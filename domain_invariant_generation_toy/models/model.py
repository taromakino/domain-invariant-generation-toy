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
        mu = self.mu(x)
        mu = mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        mu = mu[torch.arange(batch_size), y, e, :]
        cov = self.cov(x)
        cov = cov.reshape(batch_size, N_CLASSES, N_ENVS, size_to_n_tril(2 * self.z_size))
        cov = arr_to_tril(cov[torch.arange(batch_size), y, e, :])
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
        mu_causal = self.mu_causal[e]
        mu_spurious = self.mu_spurious[y, e]
        mu = torch.hstack((mu_causal, mu_spurious))
        cov_causal = arr_to_cov(self.cov_causal[e])
        cov_spurious = arr_to_cov(self.cov_spurious[y, e])
        cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=y.device)
        cov[:, :self.z_size, :self.z_size] = cov_causal
        cov[:, self.z_size:, self.z_size:] = cov_spurious
        return D.MultivariateNormal(mu, cov)


class Model(pl.LightningModule):
    def __init__(self, dpath, seed, task, x_size, z_size, h_sizes, weight_decay, lr, lr_inference, n_steps, is_spurious):
        super().__init__()
        self.save_hyperparameters()
        self.dpath = dpath
        self.seed = seed
        self.task = task
        self.z_size = z_size
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
        self.is_spurious = is_spurious
        self.vae_params, self.q_params = [], []
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
        # q(z)
        self.q_mu = nn.Parameter(torch.full((2 * z_size,), torch.nan))
        self.q_var = nn.Parameter(torch.full((2 * z_size,), torch.nan))
        self.q_params.append(self.q_mu)
        self.q_params.append(self.q_var)
        self.causal_classifier = MLP(z_size, h_sizes, 1)
        self.spurious_classifier = MLP(2 * z_size, h_sizes, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')
        self.z_sample = []
        self.z_infer, self.y, self.x = [], [], []
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def q(self):
        return D.MultivariateNormal(self.q_mu, covariance_matrix=torch.diag(self.q_var))

    def train_vae(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        z_c, z_s = torch.chunk(z, 2, dim=1)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # E_q(z_c|x,y,e)[log p(y|z_c)]
        y_pred = self.vae_classifier(z_c.detach()).view(-1)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        return log_prob_x_z, log_prob_y_zc, kl

    def classify(self, z, y):
        if self.is_spurious:
            y_pred = self.spurious_classifier(z).view(-1)
        else:
            z_c, z_s = torch.chunk(z, 2, dim=1)
            y_pred = self.causal_classifier(z_c).view(-1)
        log_prob_y_z = -F.binary_cross_entropy_with_logits(y_pred, y.float())
        return y_pred, log_prob_y_z

    def training_step(self, batch, batch_idx):
        if self.task == Task.VAE:
            x, y, e, c, s = batch
            log_prob_x_z, log_prob_y_zc, kl = self.train_vae(x, y, e)
            loss = -log_prob_x_z - log_prob_y_zc + kl
            return loss
        else:
            assert self.task == Task.CLASSIFY
            z, y, x = batch
            y_pred, log_prob_y_z = self.classify(z, y)
            loss = -log_prob_y_z
            return loss

    def validation_step(self, batch, batch_idx):
        if self.task == Task.VAE:
            x, y, e, c, s = batch
            log_prob_x_z, log_prob_y_zc, kl = self.train_vae(x, y, e)
            loss = -log_prob_x_z - log_prob_y_zc + kl
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        else:
            assert self.task == Task.CLASSIFY
            z, y, x = batch
            y_pred, log_prob_y_z = self.classify(z, y)
            loss = -log_prob_y_z
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            self.val_acc.update(y_pred, y.long())
            
    def on_validation_epoch_end(self):
        if self.task == Task.CLASSIFY:
            self.log('val_acc', self.val_acc.compute())

    def e_invariant_loss(self, x, z, q):
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.vae_classifier(z_c.detach())
        prob_y_pos_zc = torch.sigmoid(y_pred)
        prob_y_neg_zc = 1 - prob_y_pos_zc
        prob_y_zc = torch.hstack((prob_y_neg_zc, prob_y_pos_zc))
        log_prob_y_zc = torch.log(prob_y_zc.max(dim=1).values).mean()
        log_prob_z = q.log_prob(z).mean()
        return log_prob_x_z, log_prob_y_zc, log_prob_z

    def infer_z(self, x):
        batch_size = len(x)
        q = self.q()
        z_sample = q.sample((batch_size,))
        z_param = nn.Parameter(z_sample)
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_y_zc = optim_log_prob_z = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_y_zc, log_prob_z = self.e_invariant_loss(x, z_param, q)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_z
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_y_zc = log_prob_y_zc
                optim_log_prob_z = log_prob_z
                optim_z = z_param.clone()
        return optim_z, optim_log_prob_x_z, optim_log_prob_y_zc, optim_log_prob_z, optim_loss

    def test_step(self, batch, batch_idx):
        if self.task == Task.Q_Z:
            x, y, e, c, s = batch
            posterior_dist = self.encoder(x, y, e)
            z = self.sample_z(posterior_dist)
            self.z_sample.append(z.detach().cpu())
        elif self.task == Task.INFER_Z:
            x, y, e, c, s = batch
            with torch.set_grad_enabled(True):
                z, log_prob_x_z, log_prob_y_zc, log_prob_z, loss = self.infer_z(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
                self.log('log_prob_z', log_prob_z, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.z_infer.append(z.detach().cpu())
                self.y.append(y.cpu())
                self.x.append(x.cpu())
        else:
            assert self.task == Task.CLASSIFY
            z, y, x = batch
            y_pred, log_prob_y_z = self.classify(z, y)
            self.test_acc.update(y_pred, y.long())

    def on_test_epoch_end(self):
        if self.task == Task.Q_Z:
            z = torch.cat(self.z_sample)
            self.q_mu.data = torch.mean(z, 0)
            self.q_var.data = torch.var(z, 0)
        elif self.task == Task.INFER_Z:
            z, y, x = torch.cat(self.z_infer), torch.cat(self.y), torch.cat(self.x)
            torch.save((z, y, x), os.path.join(self.dpath, f'version_{self.seed}', 'z.pt'))
        else:
            assert self.task == Task.CLASSIFY
            self.log('test_acc', self.test_acc.compute())

    def configure_grad(self):
        for params in self.q_params:
            params.requires_grad = False
        if self.task == Task.VAE:
            for params in self.vae_params:
                params.requires_grad = True
            for params in self.causal_classifier.parameters():
                params.requires_grad = False
            for params in self.spurious_classifier.parameters():
                params.requires_grad = False
        elif self.task == Task.Q_Z:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.causal_classifier.parameters():
                params.requires_grad = False
            for params in self.spurious_classifier.parameters():
                params.requires_grad = False
        elif self.task == Task.INFER_Z:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.causal_classifier.parameters():
                params.requires_grad = False
            for params in self.spurious_classifier.parameters():
                params.requires_grad = False
        else:
            assert self.task == Task.CLASSIFY
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.causal_classifier.parameters():
                params.requires_grad = not self.is_spurious
            for params in self.spurious_classifier.parameters():
                params.requires_grad = self.is_spurious

    def configure_optimizers(self):
        if self.task == Task.VAE:
            return Adam(self.vae_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY:
            if self.is_spurious:
                return Adam(self.spurious_classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
            else:
                return Adam(self.causal_classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)