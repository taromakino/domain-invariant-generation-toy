import numpy as np
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, size_to_n_tril, arr_to_scale_tril, arr_to_cov
from torchmetrics import Accuracy
from data import N_CLASSES, N_ENVS


class VAE(pl.LightningModule):
    def __init__(self, stage, x_size, z_size, h_sizes, alpha_train, alpha_inference, posterior_reg_mult, q_reg_mult,
            n_components, lr, lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage
        self.z_size = z_size
        self.alpha_train = alpha_train
        self.alpha_inference = alpha_inference
        self.posterior_reg_mult = posterior_reg_mult
        self.q_reg_mult = q_reg_mult
        self.n_components = n_components
        self.lr = lr
        self.lr_inference = lr_inference
        self.n_steps = n_steps
        # q(z_c|x,y,e)
        self.encoder_mu = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * 2 * self.z_size, nn.LeakyReLU)
        self.encoder_cov = MLP(x_size, h_sizes, N_CLASSES * N_ENVS * size_to_n_tril(2 * self.z_size), nn.LeakyReLU)
        # p(x|z_c, z_s)
        self.decoder = MLP(2 * z_size, h_sizes, x_size, nn.LeakyReLU)
        # p(y|z_c)
        self.causal_classifier = MLP(z_size, h_sizes, 1, nn.LeakyReLU)
        # p(z_c|e)
        self.prior_mu_causal = nn.Parameter(torch.zeros(N_ENVS, self.z_size))
        self.prior_cov_causal = nn.Parameter(torch.zeros(N_ENVS, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_causal)
        nn.init.xavier_normal_(self.prior_cov_causal)
        # p(z_s|y,e)
        self.prior_mu_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, self.z_size))
        self.prior_cov_spurious = nn.Parameter(torch.zeros(N_CLASSES, N_ENVS, size_to_n_tril(self.z_size)))
        nn.init.xavier_normal_(self.prior_mu_spurious)
        nn.init.xavier_normal_(self.prior_cov_spurious)
        # q(z)
        self.q_logits = nn.Parameter(torch.ones(n_components))
        self.q_mu = nn.Parameter(torch.zeros(n_components, 2 * self.z_size))
        self.q_cov = nn.Parameter(torch.zeros(n_components, size_to_n_tril(2 * self.z_size)))
        nn.init.xavier_normal_(self.q_mu)
        nn.init.xavier_normal_(self.q_cov)
        self.acc = Accuracy('binary')

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        y_idx = y.int()[:, 0]
        e_idx = e.int()[:, 0]
        # z_c,z_s ~ q(z_c,z_s|x,y,e)
        posterior_dist = self.posterior_dist(x, y_idx, e_idx)
        if self.stage == 'train':
            z = self.sample_z(posterior_dist)
            # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
            x_pred = self.decoder(z)
            log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1).mean()
            # E_q(z_c|x,y,e)[log p(y|z_c)]
            z_c, z_s = torch.chunk(z, 2, dim=1)
            y_pred = self.causal_classifier(z_c)
            log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
            # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
            prior_dist = self.prior_dist(y_idx, e_idx)
            kl = D.kl_divergence(posterior_dist, prior_dist).mean()
            posterior_reg = self.posterior_reg(posterior_dist).mean()
            return log_prob_x_z, log_prob_y_zc, kl, posterior_reg
        elif self.stage == 'train_q':
            z = posterior_dist.loc
            q_dist = self.q_dist()
            loss = -q_dist.log_prob(z).mean()
            return loss
        else:
            raise ValueError

    def posterior_dist(self, x, y_idx, e_idx):
        batch_size = len(x)
        posterior_mu = self.encoder_mu(x)
        posterior_mu = posterior_mu.reshape(batch_size, N_CLASSES, N_ENVS, 2 * self.z_size)
        posterior_mu = posterior_mu[torch.arange(batch_size), y_idx, e_idx, :]
        posterior_cov = self.encoder_cov(x)
        posterior_cov = posterior_cov.reshape(batch_size, N_CLASSES, N_ENVS, size_to_n_tril(2 * self.z_size))
        posterior_cov = arr_to_scale_tril(posterior_cov[torch.arange(batch_size), y_idx, e_idx, :])
        return D.MultivariateNormal(posterior_mu, scale_tril=posterior_cov)

    def prior_dist(self, y_idx, e_idx):
        batch_size = len(y_idx)
        prior_mu_causal = self.prior_mu_causal[e_idx]
        prior_mu_spurious = self.prior_mu_spurious[y_idx, e_idx]
        prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
        prior_cov_causal = arr_to_cov(self.prior_cov_causal[e_idx])
        prior_cov_spurious = arr_to_cov(self.prior_cov_spurious[y_idx, e_idx])
        prior_cov = torch.zeros(batch_size, 2 * self.z_size, 2 * self.z_size, device=self.device)
        prior_cov[:, :self.z_size, :self.z_size] = prior_cov_causal
        prior_cov[:, self.z_size:, self.z_size:] = prior_cov_spurious
        return D.MultivariateNormal(prior_mu, prior_cov)

    def q_dist(self):
        q_categorical_dist = D.Categorical(logits=self.q_logits)
        q_gaussian_dist = D.MultivariateNormal(self.q_mu, scale_tril=arr_to_scale_tril(self.q_cov))
        return D.MixtureSameFamily(q_categorical_dist, q_gaussian_dist)

    def posterior_reg(self, posterior_dist):
        batch_size = len(posterior_dist.loc)
        mu = torch.zeros_like(posterior_dist.loc).to(self.device)
        cov = torch.eye(2 * self.z_size).expand(batch_size, 2 * self.z_size, 2 * self.z_size).to(self.device)
        standard_normal = D.MultivariateNormal(mu, cov)
        return D.kl_divergence(posterior_dist, standard_normal)

    def training_step(self, batch, batch_idx):
        if self.stage == 'train':
            log_prob_x_z, log_prob_y_zc, kl, posterior_reg = self.forward(*batch)
            loss = -log_prob_x_z - self.alpha_train * log_prob_y_zc + kl + self.posterior_reg_mult * posterior_reg
            return loss
        elif self.stage == 'train_q':
            loss = self.forward(*batch)
            return loss

    def validation_step(self, batch, batch_idx):
        if self.stage == 'train':
            log_prob_x_z, log_prob_y_zc, kl, posterior_reg = self.forward(*batch)
            loss = -log_prob_x_z - self.alpha_train * log_prob_y_zc + kl + self.posterior_reg_mult * posterior_reg
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('val_kl', kl, on_step=False, on_epoch=True)
            self.log('posterior_reg', posterior_reg, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.stage == 'train_q':
            loss = self.forward(*batch)
            self.log('val_loss', loss, on_step=False, on_epoch=True)

    def inference_loss(self, x, z, q_z):
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        prob_y1_zc = torch.sigmoid(self.causal_classifier(z_c))
        prob_y0_zc = 1 - prob_y1_zc
        prob_y_zc = torch.hstack((prob_y0_zc, prob_y1_zc))
        log_prob_y_zc = torch.log(prob_y_zc.max(dim=1).values).mean()
        log_prob_z = q_z.log_prob(z[:, None, :])
        log_prob_z = (torch.logsumexp(log_prob_z, dim=1) - np.log(log_prob_z.shape[1])).mean()
        return log_prob_x_z, log_prob_y_zc, log_prob_z

    def inference(self, x):
        batch_size = len(x)
        # posterior train
        q_z = self.q_dist()
        z_param = nn.Parameter(torch.zeros(batch_size, 2 * self.z_size, device=self.device))
        nn.init.xavier_normal_(z_param)
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_y_zc = optim_log_prob_z = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_y_zc, log_prob_z = self.inference_loss(x, z_param, q_z)
            loss = -log_prob_x_z - log_prob_y_zc - self.q_reg_mult * log_prob_z
            loss.backward()
            optim.step()
            if loss < optim_loss:
                optim_loss = loss
                optim_log_prob_x_z = log_prob_x_z
                optim_log_prob_y_zc = log_prob_y_zc
                optim_log_prob_z = log_prob_z
                optim_z = z_param.clone()
        optim_zc, optim_zs = torch.chunk(optim_z, 2, dim=1)
        return self.causal_classifier(optim_zc), optim_log_prob_x_z, optim_log_prob_y_zc, optim_log_prob_z, optim_loss

    def test_step(self, batch, batch_idx):
        self.train()
        with torch.set_grad_enabled(True):
            x, y = batch
            y_pred, log_prob_x_z, log_prob_y_zc, log_prob_z, loss = self.inference(x)
            self.log('test_log_prob_x_z', log_prob_x_z, on_step=True, on_epoch=True)
            self.log('test_log_prob_y_zc', log_prob_y_zc, on_step=True, on_epoch=True)
            self.log('test_log_prob_z', log_prob_z, on_step=True, on_epoch=True)
            self.log('test_loss', loss, on_step=True, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            acc = self.acc(y_pred_class, y)
            self.log('test_acc', acc, on_step=True, on_epoch=True)

    def configure_optimizers(self):
        if self.stage == 'train':
            return Adam([self.encoder_mu, self.encoder_cov, self.decoder, self.causal_classifier, self.prior_mu_causal,
                self.prior_cov_causal, self.prior_mu_spurious, self.prior_cov_spurious], lr=self.lr)
        elif self.stage == 'train_q':
            return Adam([self.q_logits, self.q_mu, self.q_cov], lr=self.lr)