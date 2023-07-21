import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_tril, size_to_n_tril


class VAE(pl.LightningModule):
    def __init__(self, x_size, z_size, h_sizes, n_classes, n_envs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_classes = n_classes
        self.n_envs = n_envs
        self.z_size = z_size
        # q(z_c|x,y,e)q(z_s|z_c,x,y,e)
        self.encoder_mu_causal = MLP(x_size + self.z_size, h_sizes, n_classes * n_envs * self.z_size)
        self.encoder_tril_causal = MLP(x_size + self.z_size, h_sizes, n_classes * n_envs * size_to_n_tril(self.z_size))
        self.encoder_mu_spurious = MLP(x_size, h_sizes, n_classes * n_envs * self.z_size)
        self.encoder_tril_spurious = MLP(x_size, h_sizes, n_classes * n_envs * size_to_n_tril(self.z_size))
        # p(x|z_c, z_s)
        self.decoder = MLP(2 * z_size, h_sizes, x_size)
        # p(y|z_c)
        self.causal_classifier = MLP(z_size, h_sizes, 1)
        # p(z_c|e)
        self.prior_mu_causal = nn.Parameter(torch.zeros(n_envs, self.z_size))
        self.prior_tril_causal = nn.Parameter(torch.zeros(n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_causal)
        nn.init.xavier_normal_(self.prior_tril_causal)
        # p(z_s|y,e)
        self.prior_mu_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, self.z_size))
        self.prior_tril_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_spurious)
        nn.init.xavier_normal_(self.prior_tril_spurious)

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        # z_s ~ q(z_s|x,y,e)
        posterior_dist_spurious = self.posterior_dist_spurious(x, y_idx, e_idx)
        z_s = self.sample_z(posterior_dist_spurious)
        # z_c ~ q(z_c|z_s,x,y,e)
        posterior_dist_causal = self.posterior_dist_causal(x, y_idx, e_idx, z_s)
        z_c = self.sample_z(posterior_dist_causal)
        # E_q(zc,zs|x,y,e)[log p(x|zc,zs)]
        x_pred = self.decoder(torch.hstack((z_c, z_s)))
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        y_pred = self.causal_classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        # KL(q(z_c|x,y,e) || p(z_c|e)
        prior_dist_causal = self.prior_dist_causal(e_idx)
        kl_causal = D.kl_divergence(posterior_dist_causal, prior_dist_causal)
        # KL(q(z_s|z_c,x,y,e) || p(z_s|y,e)
        prior_dist_spurious = self.prior_dist_spurious(y_idx, e_idx)
        kl_spurious = D.kl_divergence(posterior_dist_spurious, prior_dist_spurious)
        elbo = log_prob_x_z + log_prob_y_zc - kl_causal - kl_spurious
        return -elbo.mean()

    def posterior_dist_causal(self, x, y_idx, e_idx, z_s):
        batch_size = len(x)
        mu = self.encoder_mu_causal(torch.hstack((x, z_s)))
        mu = mu.reshape(batch_size, self.n_classes, self.n_envs, self.z_size)
        mu = mu[torch.arange(batch_size), y_idx, e_idx, :]
        tril = self.encoder_tril_causal(torch.hstack((x, z_s)))
        tril = tril.reshape(batch_size, self.n_classes, self.n_envs, size_to_n_tril(self.z_size))
        tril = tril[torch.arange(batch_size), y_idx, e_idx, :]
        tril = arr_to_tril(tril)
        return D.MultivariateNormal(mu, scale_tril=tril)

    def posterior_dist_spurious(self, x, y_idx, e_idx):
        batch_size = len(x)
        mu = self.encoder_mu_spurious(x)
        mu = mu.reshape(batch_size, self.n_classes, self.n_envs, self.z_size)
        mu = mu[torch.arange(batch_size), y_idx, e_idx, :]
        tril = self.encoder_tril_spurious(x)
        tril = tril.reshape(batch_size, self.n_classes, self.n_envs, size_to_n_tril(self.z_size))
        tril = tril[torch.arange(batch_size), y_idx, e_idx, :]
        tril = arr_to_tril(tril)
        return D.MultivariateNormal(mu, scale_tril=tril)

    def prior_dist_causal(self, e_idx):
        mu = self.prior_mu_causal[e_idx]
        tril = arr_to_tril(self.prior_tril_causal[e_idx])
        return D.MultivariateNormal(mu, scale_tril=tril)

    def prior_dist_spurious(self, y_idx, e_idx):
        mu = self.prior_mu_spurious[y_idx, e_idx]
        tril = arr_to_tril(self.prior_tril_spurious[y_idx, e_idx])
        return D.MultivariateNormal(mu, scale_tril=tril)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)