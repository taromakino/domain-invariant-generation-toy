import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from data import N_CLASSES, N_ENVS
from torch.optim import Adam
from torchmetrics import Accuracy, R2Score
from utils.enums import Task
from utils.nn_utils import MLP, size_to_n_tril, arr_to_cholesky


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
        cov = arr_to_cholesky(cov[torch.arange(batch_size), y_idx, e_idx, :])
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


class InferenceEncoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.mu = MLP(x_size, h_sizes, 2 * z_size)
        self.cov = MLP(x_size, h_sizes, size_to_n_tril(2 * z_size))

    def forward(self, x):
        return D.MultivariateNormal(self.mu(x), scale_tril=arr_to_cholesky(self.cov(x)))


class Model(pl.LightningModule):
    def __init__(self, dpath, seed, task, x_size, z_size, h_sizes, prior_reg_mult, weight_decay, lr):
        super().__init__()
        self.save_hyperparameters()
        self.dpath = dpath
        self.seed = seed
        self.task = task
        self.z_size = z_size
        self.prior_reg_mult = prior_reg_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.vae_params = []
        self.q_params = []
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, h_sizes)
        self.vae_params += list(self.encoder.parameters())
        # p(x|z_c,z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        self.vae_params += list(self.decoder.parameters())
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size)
        self.vae_params += list(self.prior.parameters())
        # q(z|x)
        self.inference_encoder = InferenceEncoder(x_size, z_size, h_sizes)
        # p(y|z_c)
        self.classifier_y_zc = MLP(z_size, h_sizes, 1)
        # p(c|z_c)
        self.classifier_c_zc = MLP(z_size, h_sizes, 1)
        # p(s|z_c)
        self.regressor_s_zc = MLP(z_size, h_sizes, 1)
        self.z = []
        self.y = []
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

    def train_inference_encoder(self, x, y, e):
        posterior_dist = self.encoder(x, y, e)
        inference_posterior_dist = self.inference_encoder(x)
        kl = D.kl_divergence(posterior_dist, inference_posterior_dist).mean()
        return kl

    def classify_y_zc(self, x, y):
        inference_posterior_dist = self.inference_encoder(x)
        z = inference_posterior_dist.loc
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.classifier_y_zc(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
        return y_pred, log_prob_y_zc

    def classify_c_zc(self, x, c):
        inference_posterior_dist = self.inference_encoder(x)
        z = inference_posterior_dist.loc
        z_c, z_s = torch.chunk(z, 2, dim=1)
        c_pred = self.classifier_c_zc(z_c)
        log_prob_c_zc = -F.binary_cross_entropy_with_logits(c_pred, c)
        return c_pred, log_prob_c_zc

    def regress_s_zc(self, x, s):
        inference_posterior_dist = self.inference_encoder(x)
        z = inference_posterior_dist.loc
        z_c, z_s = torch.chunk(z, 2, dim=1)
        s_pred = self.regressor_s_zc(z_c)
        mse_s_zc = F.mse_loss(s_pred, s)
        return s_pred, mse_s_zc

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.TRAIN_VAE:
            log_prob_x_z, kl, prior_reg = self.train_vae(x, y, e)
            loss = -log_prob_x_z  + kl + self.prior_reg_mult * prior_reg
            return loss
        elif self.task == Task.TRAIN_INFERENCE_ENCODER:
            kl = self.train_inference_encoder(x, y, e)
            loss = kl
            return loss
        elif self.task == Task.CLASSIFY_Y_ZC:
            y_pred, log_prob_y_zc = self.classify_y_zc(x, y)
            loss = -log_prob_y_zc
            return loss
        elif self.task == Task.CLASSIFY_C_ZC:
            c_pred, log_prob_c_zc = self.classify_c_zc(x, c)
            loss = -log_prob_c_zc
            return loss
        elif self.task == Task.REGRESS_S_ZC:
            s_pred, mse_s_zc = self.regress_s_zc(x, s)
            loss = mse_s_zc
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.TRAIN_VAE:
            log_prob_x_z, kl, prior_reg = self.train_vae(x, y, e)
            loss = -log_prob_x_z + kl + self.prior_reg_mult * prior_reg
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_prior_reg', prior_reg, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.task == Task.TRAIN_INFERENCE_ENCODER:
            kl = self.train_inference_encoder(x, y, e)
            loss = kl
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.task == Task.CLASSIFY_Y_ZC:
            y_pred, log_prob_y_zc = self.classify_y_zc(x, y)
            loss = -log_prob_y_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.val_acc.update(y_pred_class, y.long())
        elif self.task == Task.CLASSIFY_C_ZC:
            c_pred, log_prob_c_zc = self.classify_y_zc(x, c)
            loss = -log_prob_c_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.val_acc.update(c_pred_class, c.long())
        elif self.task == Task.REGRESS_S_ZC:
            s_pred, mse_s_zc = self.regress_s_zc(x, s)
            loss = mse_s_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            self.val_rsq.update(s_pred, s)

    def on_validation_epoch_end(self):
        if self.task == Task.CLASSIFY_Y_ZC or self.task == Task.CLASSIFY_C_ZC:
            self.log('val_acc', self.val_acc.compute())
        elif self.task == Task.REGRESS_S_ZC:
            self.log('val_rsq', self.val_rsq.compute())

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.CLASSIFY_Y_ZC:
            y_pred, log_prob_y_zc = self.classify_y_zc(x, y)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.test_acc.update(y_pred_class, y.long())
        elif self.task == Task.CLASSIFY_C_ZC:
            c_pred, log_prob_c_zc = self.classify_y_zc(x, c)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.test_acc.update(c_pred_class, c.long())
        elif self.task == Task.REGRESS_S_ZC:
            s_pred, mse_s_zc = self.regress_s_zc(x, s)
            self.test_rsq.update(s_pred, s)

    def on_test_epoch_end(self):
        if self.task == Task.CLASSIFY_Y_ZC or self.task == Task.CLASSIFY_C_ZC:
            self.log('test_acc', self.test_acc.compute())
        elif self.task == Task.REGRESS_S_ZC:
            self.log('test_rsq', self.test_rsq.compute())

    def configure_grad(self):
        if self.task == Task.TRAIN_VAE:
            for params in self.vae_params:
                params.requires_grad = True
            for params in self.inference_encoder.parameters():
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = False
        elif self.task == Task.TRAIN_INFERENCE_ENCODER:
            for params in self.vae_params:
                params.requires_grad = False
            for params in self.inference_encoder.parameters():
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
            for params in self.inference_encoder.parameters():
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
            for params in self.inference_encoder.parameters():
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
            for params in self.inference_encoder.parameters():
                params.requires_grad = False
            for params in self.classifier_y_zc.parameters():
                params.requires_grad = False
            for params in self.classifier_c_zc.parameters():
                params.requires_grad = False
            for params in self.regressor_s_zc.parameters():
                params.requires_grad = True

    def configure_optimizers(self):
        if self.task == Task.TRAIN_VAE:
            return Adam(self.vae_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.TRAIN_INFERENCE_ENCODER:
            return Adam(self.inference_encoder.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY_Y_ZC:
            return Adam(self.classifier_y_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.CLASSIFY_C_ZC:
            return Adam(self.classifier_c_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.task == Task.REGRESS_S_ZC:
            return Adam(self.regressor_s_zc.parameters(), lr=self.lr, weight_decay=self.weight_decay)