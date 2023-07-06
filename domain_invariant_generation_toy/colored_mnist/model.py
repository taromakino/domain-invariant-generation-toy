import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class Model(pl.LightningModule):
    def __init__(self, x_size, y_size, e_size, z_size, h_sizes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        half_z_size = z_size // 2
        # q(z_c,z_s|x,y,e)
        self.encoder_mu = MLP(x_size + y_size + e_size, h_sizes, z_size, nn.ReLU)
        self.encoder_cov = MLP(x_size + y_size + e_size, h_sizes, size_to_n_tril(z_size), nn.ReLU)
        # p(x|z_c, z_s)
        self.decoder = MLP(z_size, h_sizes, x_size, nn.ReLU)
        # p(y|z_c)
        self.p_y_zc = MLP(half_z_size, h_sizes, y_size, nn.ReLU)
        # p(z_c|e)
        self.prior_mu_causal = MLP(e_size, h_sizes, half_z_size, nn.ReLU)
        self.prior_cov_causal = MLP(e_size, h_sizes, size_to_n_tril(half_z_size), nn.ReLU)
        # p(z_s|y,e)
        self.prior_mu_spurious = MLP(y_size + e_size, h_sizes, half_z_size, nn.ReLU)
        self.prior_cov_spurious = MLP(y_size + e_size, h_sizes, size_to_n_tril(half_z_size), nn.ReLU)


    def sample_z(self, mu, cov):
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(cov, epsilon).squeeze()


    def forward(self, x, y, e):
        # z_c, z_s ~ q(z_c,z_s|x,y,e)
        posterior_mu = self.encoder_mu(x, y, e)
        posterior_cov = arr_to_scale_tril(self.encoder_cov(x, y, e))
        z = self.sample_z(posterior_mu, posterior_cov)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # E_q(z_c,z_s|x,y,e)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.p_y_zc(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        # KL(q(z_c,z_s|x,u) || p(z_c|e)p(z_s|y,e))
        posterior_dist = D.MultivariateNormal(posterior_mu, scale_tril=posterior_cov)
        prior_mu_causal = self.prior_mu_causal(e)
        prior_cov_tril_causal = arr_to_scale_tril(self.prior_cov_causal(e))
        prior_cov_causal = torch.bmm(prior_cov_tril_causal, torch.transpose(prior_cov_tril_causal, 1, 2))
        prior_mu_spurious = self.prior_mu_spurious(y, e)
        prior_cov_tril_spurious = arr_to_scale_tril(self.prior_cov_spurious(y, e))
        prior_cov_spurious = torch.bmm(prior_cov_tril_spurious, torch.transpose(prior_cov_tril_spurious, 1, 2))
        prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
        batch_size, z_size = prior_mu.shape
        half_z_size = z_size // 2
        prior_cov = torch.zeros(prior_mu.shape[0], prior_mu.shape[1], prior_mu.shape[1], device=self.device)
        prior_cov[:, :half_z_size, :half_z_size] = prior_cov_causal
        prior_cov[:, half_z_size:, half_z_size:] = prior_cov_spurious
        prior_dist = D.MultivariateNormal(prior_mu, prior_cov)
        kl = D.kl_divergence(posterior_dist, prior_dist)
        elbo = log_prob_x_z + log_prob_y_zc - kl
        return -elbo.mean()


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