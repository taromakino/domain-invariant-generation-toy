import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov


PRIOR_INIT_SD = 0.01


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)
        self.low_rank = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size * 2 * rank)
        self.diag = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        mu = self.mu(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        low_rank = self.low_rank(x)
        low_rank = low_rank.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size, 2 * self.rank)
        low_rank = low_rank[torch.arange(batch_size), y, e, :]
        diag = self.diag(x)
        diag = diag.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        diag = diag[torch.arange(batch_size), y, e, :]
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
        nn.init.normal_(self.mu_causal, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.low_rank_causal, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.diag_causal, 0, PRIOR_INIT_SD)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.low_rank_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size, rank))
        self.diag_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.low_rank_spurious, 0, PRIOR_INIT_SD)
        nn.init.normal_(self.diag_spurious, 0, PRIOR_INIT_SD)

    def forward(self, y, e):
        batch_size = len(y)
        # Causal
        mu_causal = self.mu_causal[e]
        cov_causal = arr_to_cov(self.low_rank_causal[e], self.diag_causal[e])
        # Spurious
        mu_spurious = self.mu_spurious[y, e]
        cov_spurious = arr_to_cov(self.low_rank_spurious[y, e], self.diag_spurious[y, e])
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, x_size, z_size, rank, h_sizes, q_size, beta, reg_mult, lr, weight_decay, lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.q_size = q_size
        self.beta = beta
        self.reg_mult = reg_mult
        self.lr = lr
        self.weight_decay = weight_decay
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank)
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z)
        self.q_mu = nn.Parameter(torch.full((q_size, 2 * z_size), torch.nan), requires_grad=False)
        self.q_cov = nn.Parameter(torch.full((q_size, 2 * z_size, 2 * z_size), torch.nan), requires_grad=False)
        self.q_mu_samples, self.q_cov_samples = [], []
        self.x, self.y, self.e, self.z = [], [], [], []
        self.eval_metric = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
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
        prior_norm = (prior_dist.loc ** 2).mean()
        return log_prob_x_z, log_prob_y_zc, self.beta * kl, self.reg_mult * prior_norm

    def training_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - log_prob_y_zc + kl + prior_norm
        return loss

    def validation_step(self, batch, batch_idx):
        assert self.task == Task.VAE
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_y_zc, kl, prior_norm = self.loss(x, y, e)
        loss = -log_prob_x_z - log_prob_y_zc + kl + prior_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
        self.log('val_kl', kl, on_step=False, on_epoch=True)
        self.log('val_prior_norm', prior_norm, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def infer_loss(self, x, z):
        batch_size = len(x)
        q = D.MultivariateNormal(self.q_mu, scale_tril=self.q_cov)
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z)
        # log p(y|z_c)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        # log q(z)
        log_prob_z = q.log_prob(z[:, None, :])
        log_prob_z = torch.logsumexp(log_prob_z, dim=1) - np.log(self.q_size)
        loss_candidates = []
        y_candidates = []
        for y_elem in range(N_CLASSES):
            y = torch.full((batch_size,), y_elem, dtype=torch.long, device=self.device)
            log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float(), reduction='none')
            loss_candidates.append((-log_prob_x_z - log_prob_y_zc - log_prob_z)[:, None])
            y_candidates.append(y_elem)
        loss_candidates = torch.hstack(loss_candidates).min(dim=1)
        y_candidates = torch.tensor(y_candidates, device=self.device)
        idxs = loss_candidates.indices
        y_pred = y_candidates[idxs]
        return loss_candidates.values.mean(), y_pred

    def infer_z(self, x):
        batch_size = len(x)
        z_param = nn.Parameter(torch.repeat_interleave(self.q_mu.mean(dim=0)[None], batch_size, dim=0))
        optim = Adam([z_param], lr=self.lr_infer)
        optim_loss = torch.inf
        optim_y_pred = None
        optim_z = None
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            loss, y_pred = self.infer_loss(x, z_param)
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_y_pred = y_pred
                optim_z = z_param.clone()
        return optim_loss, optim_y_pred, optim_z

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.Q:
            posterior_dist = self.encoder(x, y, e)
            self.q_mu_samples.append(posterior_dist.loc.detach().cpu())
            self.q_cov_samples.append(posterior_dist.scale_tril.detach().cpu())
        else:
            assert self.task == Task.CLASSIFY
            with torch.set_grad_enabled(True):
                loss, y_pred, z = self.infer_z(x)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.eval_metric.update(y_pred, y)
                self.x.append(x.cpu())
                self.y.append(y.cpu())
                self.e.append(e.cpu())
                self.z.append(z.detach().cpu())

    def on_test_epoch_end(self):
        if self.task == Task.Q:
            rng = np.random.RandomState(self.trainer.logger.version)
            q_mu, q_cov = torch.cat(self.q_mu_samples), torch.cat(self.q_cov_samples)
            idxs = rng.choice(len(q_mu), self.q_size, replace=False)
            self.q_mu.data = q_mu[idxs].to(self.device)
            self.q_cov.data = q_cov[idxs].to(self.device)
        else:
            assert self.task == Task.CLASSIFY
            self.log('eval_metric', self.eval_metric.compute())
            x = torch.cat(self.x)
            y = torch.cat(self.y)
            e = torch.cat(self.e)
            z = torch.cat(self.z)
            torch.save((x, y, e, z), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}',
                'infer.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)