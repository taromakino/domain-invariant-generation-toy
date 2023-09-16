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


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes, rank):
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


class Model(pl.LightningModule):
    def __init__(self, task, x_size, z_size, h_sizes, rank, reg_mult, weight_decay, lr, lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.z_size = z_size
        self.reg_mult = reg_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
        self.vae_params = []
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, h_sizes, rank)
        self.vae_params += list(self.encoder.parameters())
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        self.vae_params += list(self.decoder.parameters())
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, rank)
        self.vae_params += list(self.prior.parameters())
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z)
        self.q_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.q_var = nn.Parameter(torch.full((2 * z_size,), torch.nan), requires_grad=False)
        self.train_acc = Accuracy('binary')
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')
        self.z = []
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q(self):
        return D.MultivariateNormal(self.q_mu, covariance_matrix=torch.diag(self.q_var))

    def vae(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        return log_prob_x_z, kl

    def classify_train(self, x, y, e):
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        return y_pred, log_prob_y_zc

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.VAE:
            log_prob_x_z, kl = self.vae(x, y, e)
            loss = -log_prob_x_z + kl
            return loss
        else:
            assert self.task == Task.CLASSIFY_TRAIN
            y_pred, log_prob_y_zc = self.classify_train(x, y, e)
            loss = -log_prob_y_zc
            self.train_acc.update(y_pred, y.long())
            return loss

    def on_train_epoch_end(self):
        if self.task == Task.CLASSIFY_TRAIN:
            self.log('train_acc', self.train_acc.compute())

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.VAE:
            log_prob_x_z, kl = self.vae(x, y, e)
            loss = -log_prob_x_z + kl
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        else:
            assert self.task == Task.CLASSIFY_TRAIN
            y_pred, log_prob_y_zc = self.classify_train(x, y, e)
            loss = -log_prob_y_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            self.val_acc.update(y_pred, y.long())

    def on_validation_epoch_end(self):
        if self.task == Task.CLASSIFY_TRAIN:
            self.log('val_acc', self.val_acc.compute())

    def e_invariant_loss(self, x, z):
        log_prob_x_z = self.decoder(x, z).mean()
        z_norm = (z ** 2).mean()
        return log_prob_x_z, z_norm

    def classify_test(self, x):
        batch_size = len(x)
        q = self.q()
        z_sample = q.sample((batch_size,))
        z_param = nn.Parameter(z_sample)
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_z_norm = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, z_norm = self.e_invariant_loss(x, z_param)
            loss = -log_prob_x_z + self.reg_mult * z_norm
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_z_norm = z_norm
                optim_z = z_param.clone()
        z_c, z_s = torch.chunk(optim_z, 2, dim=1)
        y_pred = self.classifier(z_c).view(-1)
        return y_pred, optim_log_prob_x_z, optim_z_norm, optim_loss

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.Q_Z:
            posterior_dist = self.encoder(x, y, e)
            z = self.sample_z(posterior_dist)
            self.z.append(z.detach().cpu())
        else:
            assert self.task == Task.CLASSIFY_TEST
            with torch.set_grad_enabled(True):
                y_pred, log_prob_x_z, z_norm, loss = self.classify_test(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('z_norm', z_norm, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.test_acc.update(y_pred, y.long())

    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.z)
            self.q_mu.data = torch.mean(z, 0)
            self.q_var.data = torch.var(z, 0)
        else:
            assert self.task == Task.CLASSIFY_TEST
            self.log('test_acc', self.test_acc.compute())

    def configure_grad(self):
        if self.task == Task.VAE:
            for params in self.vae_params:
                params.requires_grad = True
            for params in self.classifier.parameters():
                params.requires_grad = False
        elif self.task == Task.Q_Z:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = False
        elif self.task == Task.CLASSIFY_TRAIN:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = True
        else:
            assert self.task == Task.CLASSIFY_TEST
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = False


    def configure_optimizers(self):
        if self.task == Task.VAE:
            return Adam(self.vae_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY_TRAIN:
            return Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)