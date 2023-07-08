import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class VAE(pl.LightningModule):
    def __init__(self, x_size, z_size, h_sizes, n_classes, n_envs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.n_classes = n_classes
        self.n_envs = n_envs
        self.z_size = z_size
        # q(z_c,z_s|x,y,e)
        self.encoder_mu_causal = MLP(x_size, h_sizes, n_classes * n_envs * self.z_size, nn.ReLU)
        self.encoder_cov_causal = MLP(x_size, h_sizes, n_classes * n_envs * size_to_n_tril(self.z_size), nn.ReLU)
        self.encoder_mu_spurious = MLP(x_size, h_sizes, n_classes * n_envs * self.z_size, nn.ReLU)
        self.encoder_cov_spurious = MLP(x_size, h_sizes, n_classes * n_envs * size_to_n_tril(self.z_size), nn.ReLU)
        # p(x|z_c, z_s)
        self.decoder = MLP(2 * z_size, h_sizes, x_size, nn.ReLU)
        # p(y|z_c)
        self.causal_classifier = MLP(self.z_size, h_sizes, 1, nn.ReLU)
        # p(z_c|e)
        self.prior_mu_causal = nn.Parameter(torch.zeros(n_envs, self.z_size))
        self.prior_cov_causal = nn.Parameter(torch.zeros(n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_causal)
        nn.init.xavier_normal_(self.prior_cov_causal)
        # p(z_s|y,e)
        self.prior_mu_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, self.z_size))
        self.prior_cov_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_spurious)
        nn.init.xavier_normal_(self.prior_cov_spurious)

    def forward(self, x, y, e):
        y_idx = y.squeeze().int()
        e_idx = e.squeeze().int()
        # z_c ~ q(z_c|x,y,e)
        posterior_dist_causal = self.posterior_dist_causal(x, y_idx, e_idx)
        z_c = posterior_dist_causal.sample()
        # z_s ~ q(z_s|x,y,e)
        posterior_dist_spurious = self.posterior_dist_spurious(x, y_idx, e_idx)
        z_s = posterior_dist_spurious.sample()
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        z = torch.cat((z_c, z_s), dim=1)
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        y_pred = self.causal_classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        # KL(q(z_c|x,y,e) || p(z_c|e)
        prior_dist_causal = self.prior_dist_causal(e_idx)
        kl_causal = D.kl_divergence(posterior_dist_causal, prior_dist_causal)
        # KL(q(z_s|x,y,e) || p(z_s|y,e)
        prior_dist_spurious = self.prior_dist_spurious(y_idx, e_idx)
        kl_spurious = D.kl_divergence(posterior_dist_spurious, prior_dist_spurious)
        elbo = log_prob_x_z + log_prob_y_zc - kl_causal - kl_spurious
        return -elbo.mean()

    def posterior_dist_causal(self, x, y_idx, e_idx):
        batch_size = len(x)
        posterior_mu_causal = self.encoder_mu_causal(x)
        posterior_mu_causal = posterior_mu_causal.reshape(batch_size, self.n_classes, self.n_envs, self.z_size)
        posterior_mu_causal = posterior_mu_causal[torch.arange(batch_size), y_idx, e_idx, :]
        posterior_cov_causal = self.encoder_cov_causal(x)
        posterior_cov_causal = posterior_cov_causal.reshape(batch_size, self.n_classes, self.n_envs,
            size_to_n_tril(self.z_size))
        posterior_cov_causal = arr_to_scale_tril(posterior_cov_causal[torch.arange(batch_size), y_idx, e_idx, :])
        return D.MultivariateNormal(posterior_mu_causal, scale_tril=posterior_cov_causal)

    def posterior_dist_spurious(self, x, y_idx, e_idx):
        batch_size = len(x)
        posterior_mu_spurious = self.encoder_mu_spurious(x)
        posterior_mu_spurious = posterior_mu_spurious.reshape(batch_size, self.n_classes, self.n_envs, self.z_size)
        posterior_mu_spurious = posterior_mu_spurious[torch.arange(batch_size), y_idx, e_idx, :]
        posterior_cov_spurious = self.encoder_cov_spurious(x)
        posterior_cov_spurious = posterior_cov_spurious.reshape(batch_size, self.n_classes, self.n_envs,
            size_to_n_tril(self.z_size))
        posterior_cov_spurious = arr_to_scale_tril(posterior_cov_spurious[torch.arange(batch_size), y_idx, e_idx, :])
        return D.MultivariateNormal(posterior_mu_spurious, scale_tril=posterior_cov_spurious)

    def prior_dist_causal(self, e_idx):
        prior_mu_causal = self.prior_mu_causal[e_idx]
        prior_cov_causal = arr_to_scale_tril(self.prior_cov_causal[e_idx])
        return D.MultivariateNormal(prior_mu_causal, scale_tril=prior_cov_causal)

    def prior_dist_spurious(self, y_idx, e_idx):
        prior_mu_spurious = self.prior_mu_spurious[y_idx, e_idx]
        prior_cov_spurious = arr_to_scale_tril(self.prior_cov_spurious[y_idx, e_idx])
        return D.MultivariateNormal(prior_mu_spurious, scale_tril=prior_cov_spurious)

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


class SpuriousClassifier(pl.LightningModule):
    def __init__(self, z_size, h_sizes, n_envs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.n_envs = n_envs
        self.lr = lr
        # p(y|z_s, e)
        self.net = MLP(z_size, h_sizes, n_envs, nn.ReLU)

    def forward(self, z_s, y, e):
        batch_size = len(z_s)
        e_idx = e.squeeze().int()
        y_pred = self.net(z_s).reshape(batch_size, self.n_envs, 1)
        y_pred = y_pred[torch.arange(batch_size), e_idx]
        return F.binary_cross_entropy_with_logits(y_pred, y)

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