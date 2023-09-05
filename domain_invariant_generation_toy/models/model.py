import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy, R2Score
from utils.enums import Task
from utils.nn_utils import MLP, size_to_n_tril, arr_to_tril


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
        self.cov_causal = nn.Parameter(torch.zeros(N_ENVS, z_size))
        nn.init.normal_(self.mu_causal, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov_causal, 0, GAUSSIAN_INIT_SD)
        # p(z_s|y,e)
        self.mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        self.cov_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, z_size))
        nn.init.normal_(self.mu_spurious, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov_spurious, 0, GAUSSIAN_INIT_SD)

    def forward(self, y, e):
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        mu_causal = self.mu_causal[e_idx]
        mu_spurious = self.mu_spurious[y_idx, e_idx]
        mu = torch.hstack((mu_causal, mu_spurious))
        cov_causal = F.softplus(self.cov_causal[e_idx])
        cov_spurious = F.softplus(self.cov_spurious[y_idx, e_idx])
        cov = torch.hstack((cov_causal, cov_spurious))
        return D.MultivariateNormal(mu, torch.diag_embed(cov))


class AggregatedPosterior(nn.Module):
    def __init__(self, z_size, n_components):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.mu = nn.Parameter(torch.zeros(n_components, z_size))
        self.cov = nn.Parameter(torch.zeros(n_components, size_to_n_tril(z_size)))
        nn.init.normal_(self.mu, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov, 0, GAUSSIAN_INIT_SD)

    def forward(self):
        mixture_dist = D.Categorical(logits=self.logits)
        component_dist = D.MultivariateNormal(self.mu, scale_tril=arr_to_tril(self.cov))
        return D.MixtureSameFamily(mixture_dist, component_dist)


class Model(pl.LightningModule):
    def __init__(self, dpath, seed, task, x_size, z_size, h_sizes, n_components, prior_reg_mult, q_mult, weight_decay,
            lr, lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.dpath = dpath
        self.seed = seed
        self.task = task
        self.z_size = z_size
        self.prior_reg_mult = prior_reg_mult
        self.q_mult = q_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
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
        # q(z_c)
        self.q_causal = AggregatedPosterior(z_size, n_components)
        self.q_params += list(self.q_causal.parameters())
        # q(z_s)
        self.q_spurious = AggregatedPosterior(z_size, n_components)
        self.q_params += list(self.q_spurious.parameters())
        # p(y|z_c)
        self.classifier_y_zc = MLP(z_size, h_sizes, 1)
        # p(c|z_c)
        self.classifier_c_zc = MLP(z_size, h_sizes, 1)
        # p(s|z_c)
        self.regressor_s_zc = MLP(z_size, h_sizes, 1)
        self.z, self.y, self.c, self.s = [], [], [], []
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')
        self.val_rsq = R2Score()
        self.test_rsq = R2Score()
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def prior_reg(self, prior_dist):
        batch_size = len(prior_dist.loc)
        mu = torch.zeros_like(prior_dist.loc).to(self.device)
        cov = torch.eye(2 * self.z_size).expand(batch_size, 2 * self.z_size, 2 * self.z_size).to(self.device)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(prior_dist, standard_normal)

    def train_vae(self, x, y, e):
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.encoder(x, y, e)
        z = self.sample_z(posterior_dist)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        log_prob_x_z = self.decoder(x, z).mean()
        # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
        prior_dist = self.prior(y, e)
        kl = D.kl_divergence(posterior_dist, prior_dist).mean()
        prior_reg = self.prior_reg(prior_dist).mean()
        return log_prob_x_z, kl, prior_reg

    def train_q(self, x, y, e):
        posterior_dist = self.encoder(x, y, e)
        z_c, z_s = torch.chunk(posterior_dist.loc, 2, dim=1)
        log_prob_zc = self.q_causal().log_prob(z_c).mean()
        log_prob_zs = self.q_spurious().log_prob(z_s).mean()
        log_prob_z = log_prob_zc + log_prob_zs
        return log_prob_z

    def classify_y_zc(self, z, y):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier_y_zc(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
        return y_pred, log_prob_y_zc

    def classify_c_zc(self, z, c):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        c_pred = self.classifier_c_zc(z_c)
        log_prob_c_zc = -F.binary_cross_entropy_with_logits(c_pred, c)
        return c_pred, log_prob_c_zc

    def regress_s_zc(self, z, s):
        z_c, z_s = torch.chunk(z, 2, dim=1)
        s_pred = self.regressor_s_zc(z_c)
        mse_s_zc = F.mse_loss(s_pred, s)
        return s_pred, mse_s_zc

    def log_prob_xz(self, x, z, q_causal, q_spurious):
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        log_prob_zc = q_causal.log_prob(z_c).mean()
        log_prob_zs = q_spurious.log_prob(z_s).mean()
        log_prob_z = log_prob_zc + log_prob_zs
        return log_prob_x_z, log_prob_z

    def infer_z(self, x):
        batch_size = len(x)
        q_causal = self.q_causal()
        q_spurious = self.q_spurious()
        zc_sample = q_causal.sample((batch_size,))
        zs_sample = q_spurious.sample((batch_size,))
        z_sample = torch.cat((zc_sample, zs_sample), dim=1)
        z_param = nn.Parameter(z_sample.to(self.device))
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_z = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_z = self.log_prob_xz(x, z_param, q_causal, q_spurious)
            loss = -log_prob_x_z - self.q_mult * log_prob_z
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_z = log_prob_z
                optim_z = z_param.clone()
        return optim_z, optim_log_prob_x_z, optim_log_prob_z, optim_loss

    def training_step(self, batch, batch_idx):
        if self.task == Task.TRAIN_VAE:
            x, y, e, c, s = batch
            log_prob_x_z, kl, prior_reg = self.train_vae(x, y, e)
            loss = -log_prob_x_z  + kl + self.prior_reg_mult * prior_reg
            return loss
        elif self.task == Task.TRAIN_Q:
            x, y, e, c, s = batch
            log_prob_z = self.train_q(x, y, e)
            loss = -log_prob_z
            return loss
        elif self.task == Task.CLASSIFY_Y_ZC:
            z, y, c, s = batch
            y_pred, log_prob_y_zc = self.classify_y_zc(z, y)
            loss = -log_prob_y_zc
            return loss
        elif self.task == Task.CLASSIFY_C_ZC:
            z, y, c, s = batch
            c_pred, log_prob_c_zc = self.classify_c_zc(z, c)
            loss = -log_prob_c_zc
            return loss
        elif self.task == Task.REGRESS_S_ZC:
            z, y, c, s = batch
            s_pred, mse_s_zc = self.regress_s_zc(z, s)
            loss = mse_s_zc
            return loss

    def validation_step(self, batch, batch_idx):
        if self.task == Task.TRAIN_VAE:
            x, y, e, c, s = batch
            log_prob_x_z, kl, prior_reg = self.train_vae(x, y, e)
            loss = -log_prob_x_z + kl + self.prior_reg_mult * prior_reg
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_prior_reg', prior_reg, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.task == Task.TRAIN_Q:
            x, y, e, c, s = batch
            log_prob_z = self.train_q(x, y, e)
            loss = -log_prob_z
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.task == Task.CLASSIFY_Y_ZC:
            z, y, c, s = batch
            y_pred, log_prob_y_zc = self.classify_y_zc(z, y)
            loss = -log_prob_y_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.val_acc.update(y_pred_class, y.long())
        elif self.task == Task.CLASSIFY_C_ZC:
            z, y, c, s = batch
            c_pred, log_prob_c_zc = self.classify_c_zc(z, c)
            loss = -log_prob_c_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.val_acc.update(c_pred_class, c.long())
        elif self.task == Task.REGRESS_S_ZC:
            z, y, c, s = batch
            s_pred, mse_s_zc = self.regress_s_zc(z, s)
            loss = mse_s_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            self.val_rsq.update(s_pred, s)

    def on_validation_epoch_end(self):
        if self.task == Task.CLASSIFY_Y_ZC or self.task == Task.CLASSIFY_C_ZC:
            self.log('val_acc', self.val_acc.compute())
        elif self.task == Task.REGRESS_S_ZC:
            self.log('val_rsq', self.val_rsq.compute())

    def test_step(self, batch, batch_idx):
        if self.task == Task.INFER_Z_TRAIN or self.task == Task.INFER_Z_VAL or self.task == Task.INFER_Z_TEST:
            x, y, e, c, s = batch
            with torch.set_grad_enabled(True):
                z, log_prob_x_z, log_prob_z, loss = self.infer_z(x)
                self.log('log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
                self.log('log_prob_z', log_prob_z, on_step=False, on_epoch=True)
                self.log('loss', loss, on_step=False, on_epoch=True)
                self.z.append(z.detach().cpu())
                self.y.append(y.cpu())
                self.c.append(c.cpu())
                self.s.append(s.cpu())
        elif self.task == Task.CLASSIFY_Y_ZC:
            z, y, c, s = batch
            y_pred, log_prob_y_zc = self.classify_y_zc(z, y)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.test_acc.update(y_pred_class, y.long())
        elif self.task == Task.CLASSIFY_C_ZC:
            z, y, c, s = batch
            c_pred, log_prob_c_zc = self.classify_y_zc(z, c)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.test_acc.update(c_pred_class, c.long())
        elif self.task == Task.REGRESS_S_ZC:
            z, y, c, s = batch
            s_pred, mse_s_zc = self.regress_s_zc(z, s)
            self.test_rsq.update(s_pred, s)

    def on_test_epoch_end(self):
        if self.task == Task.INFER_Z_TRAIN or self.task == Task.INFER_Z_VAL or self.task == Task.INFER_Z_TEST:
            z, y, c, s = torch.cat(self.z), torch.cat(self.y), torch.cat(self.c), torch.cat(self.s)
            torch.save((z, y, c, s), os.path.join(self.dpath, f'version_{self.seed}', 'infer.pt'))
        elif self.task == Task.CLASSIFY_Y_ZC or self.task == Task.CLASSIFY_C_ZC:
            self.log('test_acc', self.test_acc.compute())
        elif self.task == Task.REGRESS_S_ZC:
            self.log('test_rsq', self.test_rsq.compute())

    def configure_grad(self):
        if self.task == Task.TRAIN_VAE:
            for params in self.vae_params:
                params.requires_grad = True
            for params in self.q_params:
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False
        elif self.task == Task.TRAIN_Q:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.q_params:
                params.requires_grad = True
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False
        elif self.task == Task.CLASSIFY_Y_ZC:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.q_params:
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = True
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False
        elif self.task == Task.CLASSIFY_C_ZC:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.q_params:
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = True
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False
        elif self.task == Task.REGRESS_S_ZC:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.q_params:
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = True
        else:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.q_params:
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False

    def configure_optimizers(self):
        if self.task == Task.TRAIN_VAE:
            return Adam(self.vae_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.TRAIN_Q:
            return Adam(self.q_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY_Y_ZC:
            return Adam(self.classifier_y_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY_C_ZC:
            return Adam(self.classifier_c_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.REGRESS_S_ZC:
            return Adam(self.regressor_s_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)