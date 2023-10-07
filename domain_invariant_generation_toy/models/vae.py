import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril, arr_to_cov


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu = MLP(x_size + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)
        self.low_rank = MLP(x_size + N_CLASSES + N_ENVS, h_sizes, 2 * z_size * rank)
        self.diag = MLP(x_size + N_CLASSES + N_ENVS, h_sizes, 2 * z_size)

    def forward(self, x, y_embed, e_embed):
        batch_size = len(x)
        mu = self.mu(x, y_embed, e_embed)
        low_rank = self.low_rank(x, y_embed, e_embed)
        low_rank = low_rank.reshape(batch_size, 2 * self.z_size, self.rank)
        diag = self.diag(x, y_embed, e_embed)
        return D.MultivariateNormal(mu, scale_tril=arr_to_tril(low_rank, diag))


class Decoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, x_size)

    def forward(self, x, z):
        x_pred = self.mlp(z)
        return -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        # p(z_c|e)
        self.mu_causal = MLP(N_CLASSES, h_sizes, z_size)
        self.low_rank_causal = MLP(N_CLASSES, h_sizes, z_size * rank)
        self.diag_causal = MLP(N_CLASSES, h_sizes, z_size)
        # p(z_s|y,e)
        self.mu_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)
        self.low_rank_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size * rank)
        self.diag_spurious = MLP(N_CLASSES + N_ENVS, h_sizes, z_size)

    def forward(self, y_embed, e_embed):
        batch_size = len(y_embed)
        # Causal
        mu_causal = self.mu_causal(e_embed)
        low_rank_causal = self.low_rank_causal(e_embed)
        low_rank_causal = low_rank_causal.reshape(batch_size, self.z_size, self.rank)
        diag_causal = self.diag_causal(e_embed)
        cov_causal = arr_to_cov(low_rank_causal, diag_causal)
        # Spurious
        mu_spurious = self.mu_spurious(y_embed, e_embed)
        low_rank_spurious = self.low_rank_spurious(y_embed, e_embed)
        low_rank_spurious = low_rank_spurious.reshape(batch_size, self.z_size, self.rank)
        diag_spurious = self.diag_spurious(y_embed, e_embed)
        cov_spurious = arr_to_cov(low_rank_spurious, diag_spurious)
        # Block diagonal
        mu = torch.hstack((mu_causal, mu_spurious))
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y_embed.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class VAE(pl.LightningModule):
    def __init__(self, task, x_size, z_size, rank, h_sizes, beta, lr, reg_mult, weight_decay, alpha, lr_infer, n_infer_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.beta = beta
        self.lr = lr
        self.reg_mult = reg_mult
        self.weight_decay = weight_decay
        self.alpha = alpha
        self.lr_infer = lr_infer
        self.n_infer_steps = n_infer_steps
        self.y_embed = nn.Embedding(N_CLASSES, N_CLASSES)
        self.e_embed = nn.Embedding(N_ENVS, N_ENVS)
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank, h_sizes)
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z)
        self.q_z_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_z_var = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.z_sample = []
        self.x, self.y, self.z = [], [], []

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q_z(self):
        return D.MultivariateNormal(self.q_z_mu, covariance_matrix=torch.diag(self.q_z_var))

    def loss(self, x, y, e):
        y_embed = self.y_embed(y)
        e_embed = self.e_embed(e)
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y_embed, e_embed)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y_embed, e_embed)
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

    def make_infer_params(self, batch_size):
        z_param = nn.Parameter(torch.repeat_interleave(self.q_z().loc[None], batch_size, dim=0))
        y_embed = torch.arange(N_CLASSES, device=self.device).long()
        y_embed = self.y_embed(y_embed).mean(dim=0)
        y_embed = torch.repeat_interleave(y_embed[None], batch_size, dim=0)
        y_embed_param = nn.Parameter(y_embed)
        e_embed = torch.arange(N_ENVS, device=self.device).long()
        e_embed = self.e_embed(e_embed).mean(dim=0)
        e_embed = torch.repeat_interleave(e_embed[None], batch_size, dim=0)
        e_embed_param = nn.Parameter(e_embed)
        return z_param, y_embed_param, e_embed_param

    def invert_y_embed(self, y_embed):
        y_embed = torch.repeat_interleave(y_embed[:, None, :], N_CLASSES, dim=1)
        distance = torch.norm(y_embed - self.y_embed.weight.data[None], dim=2)
        return distance.argmin(dim=1)

    def infer_loss(self, x, z, y_embed, e_embed):
        # log p(x|z_c,z_s)
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # log p(y|z_c)
        y = self.invert_y_embed(y_embed)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # log p(z|y,e)
        prior_dist = self.prior(y_embed, e_embed)
        log_prob_z_ye = prior_dist.log_prob(z).mean()
        return z, log_prob_x_z, log_prob_y_zc, self.alpha * log_prob_z_ye

    def infer_z(self, x):
        z_param, y_embed_param, e_embed_param = self.make_infer_params(len(x))
        optim = Adam([z_param, y_embed_param, e_embed_param], lr=self.lr_infer)
        optim_loss = torch.inf
        optim_z = optim_log_prob_x_z = optim_log_prob_y_zc = optim_log_prob_z_ye = None
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            z, log_prob_x_z, log_prob_y_zc, log_prob_z_ye = self.infer_loss(x, z_param, y_embed_param, e_embed_param)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_z_ye
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_z = z_param.clone()
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_y_zc = log_prob_y_zc
                optim_log_prob_z_ye = log_prob_z_ye
                optim_loss = loss
        return optim_z, optim_log_prob_x_z, optim_log_prob_y_zc, optim_log_prob_z_ye, optim_loss

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.Q_Z:
            y_embed = self.y_embed(y)
            e_embed = self.e_embed(e)
            z = self.encoder(x, y_embed, e_embed).loc
            self.z_sample.append(z.detach().cpu())
        else:
            assert self.task == Task.INFER_Z
            with torch.set_grad_enabled(True):
                z, log_prob_x_z, log_prob_y_zc, log_prob_z_ye, loss = self.infer_z(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
                self.log('log_prob_z_ye', log_prob_z_ye, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.x.append(x.cpu())
                self.y.append(y.cpu())
                self.z.append(z.detach().cpu())

    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.z_sample)
            self.q_z_mu.data = torch.mean(z, 0)
            self.q_z_var.data = torch.var(z, 0)
        else:
            assert self.task == Task.INFER_Z
            x = torch.cat(self.x)
            y = torch.cat(self.y)
            z = torch.cat(self.z)
            torch.save((x, y, z), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}',
                'infer.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)