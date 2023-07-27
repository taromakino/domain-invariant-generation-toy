import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, size_to_n_tril, arr_to_scale_tril, arr_to_cov


class VAE(pl.LightningModule):
    def __init__(self, x_size, z_size, h_sizes, n_classes, n_envs, n_samples, prior_likelihood_mult, posterior_reg_mult,
            lr):
        super().__init__()
        self.save_hyperparameters()
        self.z_size = z_size
        self.n_classes = n_classes
        self.n_envs = n_envs
        self.n_samples = n_samples
        self.prior_likelihood_mult = prior_likelihood_mult
        self.posterior_reg_mult = posterior_reg_mult
        self.lr = lr
        # q(z_c|x,y,e)
        self.encoder_mu = MLP(x_size, h_sizes, n_classes * n_envs * 2 * self.z_size, nn.ReLU)
        self.encoder_cov = MLP(x_size, h_sizes, n_classes * n_envs * size_to_n_tril(2 * self.z_size), nn.ReLU)
        # p(x|z_c, z_s)
        self.decoder = MLP(2 * z_size, h_sizes, x_size, nn.ReLU)
        # p(y|z_c)
        self.causal_classifier = MLP(z_size, h_sizes, 1, nn.ReLU)
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

    def repeat_samples(self, x):
        batch_size, other_size = x.shape[0], x.shape[1:]
        expanded_shape = (self.n_samples, batch_size,) + other_size
        x = x.expand(*expanded_shape)
        return x.reshape(self.n_samples * batch_size, *other_size)

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        x, y, e = self.repeat_samples(x), self.repeat_samples(y), self.repeat_samples(e)
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.posterior_dist(x, y_idx, e_idx)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1).mean()
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.causal_classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior_dist(y_idx, e_idx)
        log_prob_prior = prior_dist.log_prob(z).mean()
        entropy_posterior = posterior_dist.entropy().mean()
        posterior_reg = self.posterior_reg(posterior_dist).mean()
        return log_prob_x_z, log_prob_y_zc, log_prob_prior, entropy_posterior, posterior_reg

    def posterior_dist(self, x, y_idx, e_idx):
        batch_size = len(x)
        posterior_mu = self.encoder_mu(x)
        posterior_mu = posterior_mu.reshape(batch_size, self.n_classes, self.n_envs, 2 * self.z_size)
        posterior_mu = posterior_mu[torch.arange(batch_size), y_idx, e_idx, :]
        posterior_cov = self.encoder_cov(x)
        posterior_cov = posterior_cov.reshape(batch_size, self.n_classes, self.n_envs, size_to_n_tril(2 * self.z_size))
        posterior_cov = arr_to_scale_tril(posterior_cov[torch.arange(batch_size), y_idx, e_idx, :])
        return D.MultivariateNormal(posterior_mu, scale_tril=posterior_cov)

    def prior_dist(self, y_idx, e_idx):
        batch_size = len(y_idx)
        prior_mu_causal = self.prior_mu_causal[e_idx]
        prior_mu_spurious = self.prior_mu_spurious[y_idx, e_idx]
        prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
        prior_cov_causal = arr_to_cov(self.prior_cov_causal[e_idx])
        prior_cov_spurious = arr_to_cov(self.prior_cov_spurious[y_idx, e_idx])
        prior_cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=self.device)
        prior_cov[:, :self.z_size, :self.z_size] = prior_cov_causal
        prior_cov[:, self.z_size:, self.z_size:] = prior_cov_spurious
        return D.MultivariateNormal(prior_mu, prior_cov)

    def posterior_reg(self, posterior_dist):
        batch_size = len(posterior_dist.loc)
        mu = torch.zeros_like(posterior_dist.loc).to(self.device)
        cov = torch.eye(2 * self.z_size).expand(batch_size, 2 * self.z_size, 2 * self.z_size).to(self.device)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(posterior_dist, standard_normal)

    def training_step(self, batch, batch_idx):
        log_prob_x_z, log_prob_y_zc, log_prob_prior, entropy_posterior, posterior_reg = self.forward(*batch)
        loss = -log_prob_x_z - log_prob_y_zc - self.prior_likelihood_mult * log_prob_prior - entropy_posterior + \
            self.posterior_reg_mult * posterior_reg
        return loss

    def validation_step(self, batch, batch_idx):
        log_prob_x_z, log_prob_y_zc, log_prob_prior, entropy_posterior, posterior_reg = self.forward(*batch)
        loss = -log_prob_x_z - log_prob_y_zc - log_prob_prior - entropy_posterior
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_log_prob_prior', log_prob_prior, on_step=False, on_epoch=True)
        self.log('val_entropy_posterior', entropy_posterior, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)