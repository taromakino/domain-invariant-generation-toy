import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, size_to_n_tril, arr_to_scale_tril, arr_to_cov
from torchmetrics import Accuracy
from data import N_CLASSES, N_ENVS


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
        cov = arr_to_scale_tril(cov[torch.arange(batch_size), y_idx, e_idx, :])
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


class AggregatedPosterior(nn.Module):
    def __init__(self, z_size):
        super().__init__()
        self.mu = nn.Parameter(torch.zeros(z_size))
        self.cov = nn.Parameter(torch.zeros(size_to_n_tril(z_size)))
        nn.init.normal_(self.mu, 0, GAUSSIAN_INIT_SD)
        nn.init.normal_(self.cov, 0, GAUSSIAN_INIT_SD)

    def forward(self):
        return D.MultivariateNormal(self.mu, scale_tril=arr_to_scale_tril(F.softplus(self.cov)))


class VAE(pl.LightningModule):
    def __init__(self, stage, x_size, z_size, h_sizes, prior_reg_mult, q_mult, weight_decay, lr, lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage
        self.z_size = z_size
        self.prior_reg_mult = prior_reg_mult
        self.q_mult = q_mult
        self.weight_decay = weight_decay
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
        self.train_params = []
        self.train_q_params = []
        # q(z_c|x,y,e)
        self.encoder = Encoder(x_size, z_size, h_sizes)
        self.train_params += list(self.encoder.parameters())
        # p(x|z_c, z_s)
        self.decoder = Decoder(x_size, z_size, h_sizes)
        self.train_params += list(self.decoder.parameters())
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size)
        self.train_params += list(self.prior.parameters())
        # p(y|z_c)
        self.classifier = MLP(z_size, h_sizes, 1)
        # q(z_c)
        self.q_causal = AggregatedPosterior(z_size)
        self.train_q_params += list(self.q_causal.parameters())
        # q(z_s)
        self.q_spurious = AggregatedPosterior(z_size)
        self.train_q_params += list(self.q_spurious.parameters())
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        posterior_dist = self.encoder(x, y, e)
        if self.stage == 'train_vae':
            # z_c,z_s ~ q(z_c,z_s|x,y,e)
            z = self.sample_z(posterior_dist)
            # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
            log_prob_x_z = self.decoder(x, z).mean()
            # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
            prior_dist = self.prior(y, e)
            kl = D.kl_divergence(posterior_dist, prior_dist).mean()
            prior_reg = self.prior_reg(prior_dist).mean()
            return log_prob_x_z, kl, prior_reg
        elif self.stage == 'train_classifier':
            z_c, z_s = torch.chunk(posterior_dist.loc, 2, dim=1)
            # E_q(z_c|x,y,e)[log p(y|z_c)]
            y_pred = self.classifier(z_c)
            log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
            return y_pred, log_prob_y_zc
        elif self.stage == 'train_q':
            z_c, z_s = torch.chunk(posterior_dist.loc, 2, dim=1)
            log_prob_zc = self.q_causal().log_prob(z_c).mean()
            log_prob_zs = self.q_spurious().log_prob(z_s).mean()
            log_prob_z = log_prob_zc + log_prob_zs
            return log_prob_z
        else:
            raise ValueError

    def prior_reg(self, prior_dist):
        batch_size = len(prior_dist.loc)
        mu = torch.zeros_like(prior_dist.loc).to(self.device)
        cov = torch.eye(2 * self.z_size).expand(batch_size, 2 * self.z_size, 2 * self.z_size).to(self.device)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(prior_dist, standard_normal)

    def training_step(self, batch, batch_idx):
        if self.stage == 'train_vae':
            log_prob_x_z, kl, prior_reg = self.forward(*batch)
            loss = -log_prob_x_z  + kl + self.prior_reg_mult * prior_reg
            return loss
        elif self.stage == 'train_classifier':
            y_pred, log_prob_y_zc = self.forward(*batch)
            loss = -log_prob_y_zc
            return loss
        elif self.stage == 'train_q':
            log_prob_z = self.forward(*batch)
            loss = -log_prob_z
            return loss

    def validation_step(self, batch, batch_idx):
        if self.stage == 'train_vae':
            log_prob_x_z, kl, prior_reg = self.forward(*batch)
            loss = -log_prob_x_z + kl + self.prior_reg_mult * prior_reg
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('val_prior_reg', prior_reg, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.stage == 'train_classifier':
            x, y, e = batch
            y_pred, log_prob_y_zc = self.forward(*batch)
            loss = -log_prob_y_zc
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.val_acc.update(y_pred_class, y.long())
        elif self.stage == 'train_q':
            log_prob_z = self.forward(*batch)
            loss = -log_prob_z
            self.log('val_loss', loss, on_step=False, on_epoch=True)

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def inference_loss(self, x, z, q_causal, q_spurious):
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        log_prob_zc = q_causal.log_prob(z_c).mean()
        log_prob_zs = q_spurious.log_prob(z_s).mean()
        log_prob_z = log_prob_zc + log_prob_zs
        return log_prob_x_z, log_prob_z

    def inference(self, x):
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
            log_prob_x_z, log_prob_z = self.inference_loss(x, z_param, q_causal, q_spurious)
            loss = -log_prob_x_z - self.q_mult * log_prob_z
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_z = log_prob_z
                optim_z = z_param.clone()
        optim_zc, optim_zs = torch.chunk(optim_z, 2, dim=1)
        return self.classifier(optim_zc), optim_log_prob_x_z, optim_log_prob_z, optim_loss

    def test_step(self, batch, batch_idx):
        with torch.set_grad_enabled(True):
            x, y = batch
            y_pred, log_prob_x_z, log_prob_z, loss = self.inference(x)
            self.log('test_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('test_log_prob_z', log_prob_z, on_step=False, on_epoch=True)
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.test_acc.update(y_pred_class, y.long())

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_grad(self):
        if self.stage == 'train_vae':
            for params in self.train_params:
                params.requires_grad = True
            for params in self.classifier.parameters():
                params.requires_grad = False
            for params in self.train_q_params:
                params.requires_grad = False
        elif self.stage == 'train_classifier':
            for params in self.train_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = True
            for params in self.train_q_params:
                params.requires_grad = False
        elif self.stage == 'train_q':
            for params in self.train_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = False
            for params in self.train_q_params:
                params.requires_grad = True
        else:
            for params in self.train_params:
                params.requires_grad = False
            for params in self.classifier.parameters():
                params.requires_grad = False
            for params in self.train_q_params:
                params.requires_grad = False

    def configure_optimizers(self):
        if self.stage == 'train_vae':
            return Adam(self.train_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.stage == 'train_classifier':
            return Adam(self.classifier.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        elif self.stage == 'train_q':
            return Adam(self.train_q_params, lr=self.lr, weight_decay=self.weight_decay)