import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from utils.enums import Task
from utils.nn_utils import MLP, arr_to_tril


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, rank, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.rank = rank
        self.mu = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)
        self.low_rank = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size * rank)
        self.diag = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)

    def forward(self, x, y, e):
        batch_size = len(x)
        mu = self.mu(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        low_rank = self.low_rank(x)
        low_rank = low_rank.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size, self.rank)
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


class PriorCausal(nn.Module):
    def __init__(self, z_size, rank):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(N_CLASSES, z_size))
        self.low_rank = nn.Parameter(torch.zeros(N_CLASSES, z_size, rank))
        self.diag = nn.Parameter(torch.zeros(N_CLASSES, z_size))
        nn.init.kaiming_normal_(self.mu)
        nn.init.kaiming_normal_(self.low_rank)
        nn.init.kaiming_normal_(self.diag)

    def forward(self, y):
        return D.MultivariateNormal(self.mu[y], scale_tril=arr_to_tril(self.low_rank[y], self.diag[y]))


class PriorSpurious(nn.Module):
    def __init__(self, e_size, z_size, rank):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(N_CLASSES, e_size, z_size))
        self.low_rank = nn.Parameter(torch.zeros(N_CLASSES, e_size, z_size, rank))
        self.diag = nn.Parameter(torch.zeros(N_CLASSES, e_size, z_size))
        nn.init.kaiming_normal_(self.mu)
        nn.init.kaiming_normal_(self.low_rank)
        nn.init.kaiming_normal_(self.diag)

    def forward(self, y, e):
        return D.MultivariateNormal(self.mu[y, e], scale_tril=arr_to_tril(self.low_rank[y, e], self.diag[y, e]))


class VAE(pl.LightningModule):
    def __init__(self, task, x_size, z_size, rank, h_sizes, y_mult, reg_mult, weight_decay, lr, lr_infer, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.y_mult = y_mult
        self.reg_mult = reg_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_infer
        self.n_steps = n_steps
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, rank, h_sizes)
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        # p(z_c|y)
        self.prior_causal = PriorCausal(z_size, rank)
        # p(z_s|y,e)
        self.prior_spurious = PriorSpurious(N_ENVS, z_size, rank)
        # q(z)
        self.q_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_var = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.z_sample = []
        self.z_infer, self.y, self.e = [], [], []

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q(self):
        return D.MultivariateNormal(self.q_mu, covariance_matrix=torch.diag(self.q_var))

    def vae_loss(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c,z_s|x,y,e)[log p(z_c|y)]
        log_prob_zc_y = self.prior_causal(y).log_prob(z_c).mean()
        # E_q(z_c,z_s|x,y,e)[log p(z_s|y,e)]
        log_prob_zs_ye = self.prior_spurious(y, e).log_prob(z_s).mean()
        # E_q(z_c,z_s|x,y,e)[-log q(z_c,z_s|x,y,e)]
        entropy = posterior_dist.entropy().mean()
        z_norm = (z ** 2).mean()
        return log_prob_x_z, log_prob_zc_y, log_prob_zs_ye, entropy, z_norm

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_zc_y, log_prob_zs_ye, entropy, z_norm = self.vae_loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * (log_prob_zc_y + log_prob_zs_ye) - entropy + self.reg_mult * z_norm
        return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        log_prob_x_z, log_prob_zc_y, log_prob_zs_ye, entropy, z_norm = self.vae_loss(x, y, e)
        loss = -log_prob_x_z - self.y_mult * (log_prob_zc_y + log_prob_zs_ye) - entropy + self.reg_mult * z_norm
        self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
        self.log('val_log_prob_zc_y', self.y_mult * log_prob_zc_y, on_step=False, on_epoch=True)
        self.log('val_log_prob_zs_ye', self.y_mult * log_prob_zs_ye, on_step=False, on_epoch=True)
        self.log('val_entropy', entropy, on_step=False, on_epoch=True)
        self.log('val_z_norm', self.reg_mult * z_norm, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def e_invariant_loss(self, x, z, q):
        batch_size = len(x)
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        log_prob_zc_y0 = self.prior_causal(torch.zeros(batch_size, device=self.device)).log_prob(z_c)
        log_prob_zc_y1 = self.prior_causal(torch.ones(batch_size, device=self.device)).log_prob(z_c)
        log_prob_zc_y = torch.hstack((log_prob_zc_y0, log_prob_zc_y1)).max(dim=1).mean()
        log_prob_z = q.log_prob(z).mean()
        return log_prob_x_z, log_prob_zc_y, log_prob_z

    def infer_z(self, x):
        batch_size = len(x)
        q = self.q()
        z_sample = q.sample((batch_size,))
        z_param = nn.Parameter(z_sample)
        optim = Adam([z_param], lr=self.lr_infer)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_zc_y = optim_log_prob_z = optim_z = None
        for _ in range(self.n_infer_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_zc_y, log_prob_z = self.e_invariant_loss(x, z_param, q)
            loss = -log_prob_x_z - self.y_mult * log_prob_zc_y - log_prob_z
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_zc_y = self.y_mult * log_prob_zc_y
                optim_log_prob_z = log_prob_z
                optim_z = z_param.clone()
        return optim_z, optim_log_prob_x_z, optim_log_prob_zc_y, optim_log_prob_z, optim_loss

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.Q_Z:
            z = self.encoder(x).loc
            self.z_sample.append(z.detach().cpu())
        else:
            assert self.task == Task.INFER_Z
            self.decoder.gru.train()
            with torch.set_grad_enabled(True):
                z, log_prob_x_z, log_prob_zc_y, log_prob_z, loss = self.infer_z(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('log_prob_zc_y', log_prob_zc_y, on_step=False, on_epoch=True)
                self.log('log_prob_z', log_prob_z, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.z_infer.append(z.detach().cpu())
                self.y.append(y.cpu())
                self.e.append(e.cpu())

    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.z)
            self.q_mu.data = torch.mean(z, 0)
            self.q_var.data = torch.var(z, 0)
        else:
            assert self.task == Task.INFER_Z
            z = torch.cat(self.z_infer)
            y = torch.cat(self.y)
            e = torch.cat(self.e)
            torch.save((z, y, e), os.path.join(self.trainer.log_dir, f'version_{self.trainer.logger.version}', 'infer_z.pt'))

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)