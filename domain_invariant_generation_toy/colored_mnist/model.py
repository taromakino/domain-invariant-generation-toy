import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class VAE(pl.LightningModule):
    def __init__(self, x_size, z_size, h_sizes, n_classes, n_envs, prior_reg_mult, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.prior_reg_mult = prior_reg_mult
        self.n_classes = n_classes
        self.n_envs = n_envs
        self.z_size = z_size
        # q(z_c|x,e)
        self.encoder_mu = MLP(x_size, h_sizes, n_envs * 2 * self.z_size, nn.ReLU)
        self.encoder_cov_tril = MLP(x_size, h_sizes, n_envs * size_to_n_tril(2 * self.z_size), nn.ReLU)
        # p(x|z_c, z_s)
        self.decoder = MLP(2 * z_size, h_sizes, x_size, nn.ReLU)
        # p(y|z_c)
        self.causal_classifier = MLP(z_size, h_sizes, 1, nn.ReLU)
        # p(z_c|e)
        self.prior_mu_causal = nn.Parameter(torch.zeros(n_envs, self.z_size))
        self.prior_cov_tril_causal = nn.Parameter(torch.zeros(n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_causal)
        nn.init.xavier_normal_(self.prior_cov_tril_causal)
        # p(z_s|y,e)
        self.prior_mu_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, self.z_size))
        self.prior_cov_tril_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_spurious)
        nn.init.xavier_normal_(self.prior_cov_tril_spurious)

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.posterior_dist(x, e_idx)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.causal_classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior_dist(y_idx, e_idx)
        kl = D.kl_divergence(posterior_dist, prior_dist)
        elbo = log_prob_x_z + log_prob_y_zc - kl
        return -elbo.mean() + self.prior_reg_mult * torch.norm(prior_dist.loc)

    def posterior_dist(self, x, e_idx):
        batch_size = len(x)
        posterior_mu = self.encoder_mu(x)
        posterior_mu = posterior_mu.reshape(batch_size, self.n_envs, 2 * self.z_size)
        posterior_mu = posterior_mu[torch.arange(batch_size), e_idx, :]
        posterior_cov_tril_causal = self.encoder_cov_tril(x)
        posterior_cov_tril_causal = posterior_cov_tril_causal.reshape(batch_size, self.n_envs, size_to_n_tril(2 * self.z_size))
        posterior_cov_tril_causal = arr_to_scale_tril(posterior_cov_tril_causal[torch.arange(batch_size), e_idx, :])
        return D.MultivariateNormal(posterior_mu, scale_tril=posterior_cov_tril_causal)

    def prior_dist(self, y_idx, e_idx):
        batch_size = len(y_idx)
        prior_mu_causal = self.prior_mu_causal[e_idx]
        prior_mu_spurious = self.prior_mu_spurious[y_idx, e_idx]
        prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
        prior_cov_tril_causal = arr_to_scale_tril(self.prior_cov_tril_causal[e_idx])
        prior_cov_causal = torch.bmm(prior_cov_tril_causal, torch.transpose(prior_cov_tril_causal, 1, 2))
        prior_cov_tril_spurious = arr_to_scale_tril(self.prior_cov_tril_spurious[y_idx, e_idx])
        prior_cov_spurious = torch.bmm(prior_cov_tril_spurious, torch.transpose(prior_cov_tril_spurious, 1, 2))
        prior_cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=self.device)
        prior_cov[:, :self.z_size, :self.z_size] = prior_cov_causal
        prior_cov[:, self.z_size:, self.z_size:] = prior_cov_spurious
        return D.MultivariateNormal(prior_mu, prior_cov)

    def prior_reg(self, prior_dist):
        batch_size = len(prior_dist.loc)
        mu = torch.zeros_like(prior_dist.loc).to(self.device)
        cov = torch.eye(2 * self.z_size).expand(batch_size, 2 * self.z_size, 2 * self.z_size).to(self.device)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(prior_dist, standard_normal)

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