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
from utils.nn_utils import MLP, size_to_n_tril, arr_to_tril, arr_to_cov


GAUSSIAN_INIT_SD = 0.1


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.mu = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * z_size)
        self.cov = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * size_to_n_tril(2 * z_size))

    def forward(self, x, y, e):
        batch_size = len(x)
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        mu = self.mu(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        mu = mu[torch.arange(batch_size), y_idx, e_idx, :]
        cov = self.cov(x)
        cov = cov.reshape(batch_size, N_CLASSES, N_ENVS, size_to_n_tril(2 * self.z_size))
        cov = arr_to_tril(cov[torch.arange(batch_size), y_idx, e_idx, :])
        return D.MultivariateNormal(mu, scale_tril=cov)


class Decoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, x_size)

    def forward(self, x, z):
        x_pred = self.mlp(z)
        return -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.z_size = z_size
        self.mu_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        self.cov_causal = nn.Parameter(torch.zeros(N_ENVS, size_to_n_tril(z_size)))
        nn.init.normal_(self.mu_causal, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov_causal, 0, GAUSSIAN_INIT_SD)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.cov_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, size_to_n_tril(z_size)))
        nn.init.normal_(self.mu_spurious, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov_spurious, 0, GAUSSIAN_INIT_SD)

    def forward(self, y, e):
        batch_size = len(y)
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        mu_causal = self.mu_causal[e_idx]
        mu_spurious = self.mu_spurious[y_idx, e_idx]
        mu = torch.hstack((mu_causal, mu_spurious))
        cov_causal = arr_to_cov(self.cov_causal[e_idx])
        cov_spurious = arr_to_cov(self.cov_spurious[y_idx, e_idx])
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class Model(pl.LightningModule):
    def __init__(self, dpath, seed, task, x_size, z_size, h_sizes, z_norm_mult, weight_decay, lr, lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.dpath = dpath
        self.seed = seed
        self.task = task
        self.z_size = z_size
        self.z_norm_mult = z_norm_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
        self.vae_params = []
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, h_sizes)
        self.vae_params += list(self.encoder.parameters())
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        self.vae_params += list(self.decoder.parameters())
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size)
        self.vae_params += list(self.prior.parameters())
        # p(y|z_c)
        self.vae_classifier = MLP(z_size, h_sizes, 1)
        self.vae_params += list(self.vae_classifier.parameters())
        self.classifier = MLP(z_size, h_sizes, 1)
        self.z_c, self.y = [], []
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def train_vae(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        y_pred = self.vae_classifier(z_c.detach())
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        z_norm = (z ** 2).sum().mean()
        return log_prob_x_z, log_prob_y_zc, kl, z_norm

    def classify(self, x, y, e):
        posterior_dist = self.encoder(x, y, e)
        z = posterior_dist.loc
        # z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
        return y_pred, log_prob_y_zc

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.TRAIN_VAE:
            log_prob_x_z, log_prob_y_zc, kl, z_norm = self.train_vae(x, y, e)
            loss = -log_prob_x_z - log_prob_y_zc + kl + self.z_norm_mult * z_norm
            return loss
        elif self.task == Task.TRAIN_CLASSIFIER:
            y_pred, log_prob_y_zc = self.classify(x, y, e)
            loss = -log_prob_y_zc
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.TRAIN_VAE:
            log_prob_x_z, log_prob_y_zc, kl, z_norm = self.train_vae(x, y, e)
            loss = -log_prob_x_z - log_prob_y_zc + kl + self.z_norm_mult * z_norm
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_z_norm', z_norm, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.task == Task.TRAIN_CLASSIFIER:
            y_pred, log_prob_y_zc = self.classify(x, y, e)
            loss = -log_prob_y_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.val_acc.update(y_pred_class, y.long())

    def on_validation_epoch_end(self):
        if self.task == Task.TRAIN_CLASSIFIER:
            self.log('val_acc', self.val_acc.compute())

    def inference_loss(self, x, z):
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        prob_y_pos_zc = torch.sigmoid(self.classifier(z_c))
        prob_y_neg_zc = 1 - prob_y_pos_zc
        prob_y_zc = torch.hstack((prob_y_neg_zc, prob_y_pos_zc))
        log_prob_y_zc = torch.log(prob_y_zc.max(dim=1).values).mean()
        z_norm = (z ** 2).sum().mean()
        return log_prob_x_z, log_prob_y_zc, z_norm

    def inference(self, x):
        batch_size = len(x)
        z_param = nn.Parameter(torch.zeros(batch_size, 2 * self.z_size).to(self.device))
        nn.init.normal_(z_param, 0, GAUSSIAN_INIT_SD)
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_y_zc = optim_z_norm = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_y_zc, z_norm = self.inference_loss(x, z_param)
            loss = -log_prob_x_z - log_prob_y_zc + self.z_norm_mult * z_norm
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_y_zc = log_prob_y_zc
                optim_z_norm = z_norm
                optim_z = z_param.clone()
        optim_zc, optim_zs = torch.chunk(optim_z, 2, dim=1)
        return self.classifier(optim_zc), optim_log_prob_x_z, optim_log_prob_y_zc, optim_z_norm, optim_loss

    def test_step(self, batch, batch_idx):
        assert self.task == Task.INFERENCE
        x, y, e, c, s = batch
        with torch.set_grad_enabled(True):
            y_pred, log_prob_x_z, log_prob_y_zc, z_norm, loss = self.inference(x)
            self.log('test_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('test_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('test_z_norm', z_norm, on_step=False, on_epoch=True)
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.test_acc.update(y_pred_class, y.long())

    def on_test_epoch_end(self):
        assert self.task == Task.INFERENCE
        self.log('test_acc', self.test_acc.compute())

    def configure_grad(self):
        if self.task == Task.TRAIN_VAE:
            for params in self.vae_params:
                params.requires_grad = True
            for params in self.classifier.parameters():
                params.requires_grad = False
        elif self.task == Task.TRAIN_CLASSIFIER:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = True
        else:
            assert self.task == Task.INFERENCE
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = False

    def configure_optimizers(self):
        if self.task == Task.TRAIN_VAE:
            return Adam(self.vae_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.TRAIN_CLASSIFIER:
            return Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)