import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu = MLP(x_size, h_sizes, 2 * z_size)
        self.low_rank = MLP(x_size, h_sizes, 2 * z_size * rank)
        self.diag = MLP(x_size, h_sizes, 2 * z_size)

    def forward(self, x):
        batch_size = len(x)
        mu = self.mu(x)
        low_rank = self.low_rank(x)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, self.rank)
        diag = self.diag(x)
        return D.MultivariateNormal(mu, scale_tril=arr_to_tril(low_rank, diag))


class Decoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, x_size)

    def forward(self, x, z):
        x_pred = self.mlp(z)
        return -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.low_rank_causal = nn.Parameter(torch.zeros(N_ENVS, z_size, rank))
        self.diag_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.kaiming_normal_(self.mu_causal)
        nn.init.kaiming_normal_(self.low_rank_causal)
        nn.init.kaiming_normal_(self.diag_causal)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.kaiming_normal_(self.mu_spurious)
        nn.init.kaiming_normal_(self.low_rank_spurious)
        nn.init.kaiming_normal_(self.diag_spurious)

    def forward(self, y, e):
        batch_size = len(y)
        mu_causal = self.mu_causal[e]
        mu_spurious = self.mu_spurious[y, e]
        mu = torch.hstack((mu_causal, mu_spurious))
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, x_size, z_size, rank, h_sizes, y_mult, reg_mult, weight_decay, lr):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.reg_mult = reg_mult
        self.weight_decay = weight_decay
        self.lr = lr
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank)
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        self.z_sample = []
        self.z, self.y, self.e = [], [], []

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q(self):
        return D.MultivariateNormal(self.q_mu, covariance_matrix=torch.diag(self.q_var))

    def vae_loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        z_norm = (z ** 2).mean()
        return log_prob_x_z, log_prob_y_zc, kl, z_norm

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, z_norm = self.vae_loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + kl + self.reg_mult * z_norm
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, z_norm = self.vae_loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * log_prob_y_zc + kl + self.reg_mult * z_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True)
        self.log('val_z_norm', z_norm, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        z = self.encoder(x).loc
        self.z.append(z.detach().cpu())
        self.y.append(y.cpu())
        self.e.append(e.cpu())

    def on_test_epoch_end(self):
        z = torch.cat(self.z)
        y = torch.cat(self.y)
        e = torch.cat(self.e)
        torch.save((z, y, e), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}', 'infer_z.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)