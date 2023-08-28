import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, size_to_n_tril, arr_to_scale_tril, arr_to_cov
from torchmetrics import Accuracy


class Encoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.z_size = z_size
        self.mu = MLP(x_size + 1 + 1, h_sizes, 2 * z_size, nn.LeakyReLU)
        self.cov = MLP(x_size + 1 + 1, h_sizes, size_to_n_tril(2 * z_size), nn.LeakyReLU)

    def forward(self, x, y, e):
        mu = self.mu(x, y, e)
        cov = arr_to_scale_tril(self.cov(x, y, e))
        return D.MultivariateNormal(mu, scale_tril=cov)


class Decoder(nn.Module):
    def __init__(self, x_size, z_size, h_sizes):
        super().__init__()
        self.mlp = MLP(2 * z_size, h_sizes, x_size, nn.LeakyReLU)

    def forward(self, x, z):
        x_pred = self.mlp(z)
        return -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)


class Prior(nn.Module):
    def __init__(self, z_size, h_sizes):
        super().__init__()
        self.z_size = z_size
        # p(z_c|e)
        self.mu_causal = MLP(1, h_sizes, z_size, nn.LeakyReLU)
        self.cov_causal = MLP(1, h_sizes, size_to_n_tril(z_size), nn.LeakyReLU)
        # p(z_s|y,e)
        self.mu_spurious = MLP(1 + 1, h_sizes, z_size, nn.LeakyReLU)
        self.cov_spurious = MLP(1 + 1, h_sizes, size_to_n_tril(z_size), nn.LeakyReLU)

    def forward(self, y, e):
        mu_causal = self.mu_causal(e)
        cov_causal = arr_to_scale_tril(self.cov_causal(e))
        dist_causal = D.MultivariateNormal(mu_causal, scale_tril=cov_causal)
        mu_spurious = self.mu_spurious(y, e)
        cov_spurious = arr_to_scale_tril(self.cov_spurious(y, e))
        dist_spurious = D.MultivariateNormal(mu_spurious, scale_tril=cov_spurious)
        return dist_causal, dist_spurious


class AggregatedPosterior(nn.Module):
    def __init__(self, z_size, n_components):
        super().__init__()
        self.logits = nn.Parameter(torch.ones(n_components))
        self.mu = nn.Parameter(torch.zeros(n_components, z_size))
        self.cov = nn.Parameter(torch.zeros(n_components, size_to_n_tril(z_size)))
        nn.init.xavier_normal_(self.mu)
        nn.init.xavier_normal_(self.cov)

    def forward(self, z):
        mixture_dist = D.Categorical(logits=self.logits)
        component_dist = D.MultivariateNormal(self.mu, scale_tril=arr_to_scale_tril(self.cov))
        dist = D.MixtureSameFamily(mixture_dist, component_dist)
        return dist.log_prob(z)


class VAE(pl.LightningModule):
    def __init__(self, stage, x_size, z_size, h_sizes, n_components, y_mult, prior_mult, q_mult, weight_decay, lr,
            lr_inference, n_steps):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage
        self.z_size = z_size
        self.y_mult = y_mult
        self.prior_mult = prior_mult
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
        # p(y|z_c)
        self.causal_classifier = MLP(z_size, h_sizes, 1, nn.LeakyReLU)
        self.train_params += list(self.causal_classifier.parameters())
        # p(z_c,z_s|y,e)
        self.prior = Prior(z_size, h_sizes)
        self.train_params += list(self.prior.parameters())
        # q(z_c)
        self.q_causal = AggregatedPosterior(z_size, n_components)
        self.train_q_params += list(self.q_causal.parameters())
        # q(z_s)
        self.q_spurious = AggregatedPosterior(z_size, n_components)
        self.train_q_params += list(self.q_spurious.parameters())
        self.acc = Accuracy('binary')
        self.configure_grad()

    def sample_z(self, dist):
        mu, scale_tril = dist.loc, dist.scale_tril
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(scale_tril, epsilon).squeeze()

    def forward(self, x, y, e):
        if self.stage == 'train':
            # z_c,z_s ~ q(z_c,z_s|x,y,e)
            posterior_dist = self.encoder(x, y, e)
            z = self.sample_z(posterior_dist)
            z_c, z_s = torch.chunk(z, 2, dim=1)
            # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
            log_prob_x_z = self.decoder(x, z).mean()
            # E_q(z_c|x,y,e)[log p(y|z_c)]
            y_pred = self.causal_classifier(z_c)
            log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
            # KL(q(z_c,z_s|x,y,e) || p(z_c|e)p(z_s|y,e))
            prior_dist_causal, prior_dist_spurious = self.prior(y, e)
            log_prob_zc_e = prior_dist_causal.log_prob(z_c).mean()
            log_prob_zs_ye = prior_dist_spurious.log_prob(z_s).mean()
            log_prob_prior = log_prob_zc_e + log_prob_zs_ye
            entropy = posterior_dist.entropy().mean()
            return log_prob_x_z, self.y_mult * log_prob_y_zc, self.prior_mult * log_prob_prior, entropy
        elif self.stage == 'train_q':
            posterior_dist = self.encoder(x, y, e)
            z_c, z_s = torch.chunk(posterior_dist.loc, 2, dim=1)
            log_prob_zc = self.q_causal(z_c).mean()
            log_prob_zs = self.q_spurious(z_s).mean()
            log_prob_z = log_prob_zc + log_prob_zs
            return log_prob_z
        else:
            raise ValueError

    def training_step(self, batch, batch_idx):
        if self.stage == 'train':
            log_prob_x_z, log_prob_y_zc, log_prob_prior, entropy = self.forward(*batch)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_prior - entropy
            return loss
        elif self.stage == 'train_q':
            log_prob_z = self.forward(*batch)
            loss = -log_prob_z
            return loss

    def validation_step(self, batch, batch_idx):
        if self.stage == 'train':
            log_prob_x_z, log_prob_y_zc, log_prob_prior, entropy = self.forward(*batch)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_prior - entropy
            self.log('val_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('val_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('val_log_prior', log_prob_prior, on_step=False, on_epoch=True)
            self.log('val_entropy', entropy, on_step=False, on_epoch=True)
            self.log('val_loss', loss, on_step=False, on_epoch=True)
        elif self.stage == 'train_q':
            log_prob_z = self.forward(*batch)
            loss = -log_prob_z
            self.log('val_loss', loss, on_step=False, on_epoch=True)

    def inference_loss(self, x, z):
        log_prob_x_z = self.decoder(x, z).mean()
        z_c, z_s = torch.chunk(z, 2, dim=1)
        prob_y_pos_zc = torch.sigmoid(self.causal_classifier(z_c))
        prob_y_neg_zc = 1 - prob_y_pos_zc
        prob_y_zc = torch.hstack((prob_y_neg_zc, prob_y_pos_zc))
        log_prob_y_zc = torch.log(prob_y_zc.max(dim=1).values).mean()
        log_prob_zc = self.q_causal(z_c).mean()
        log_prob_zs = self.q_spurious(z_s).mean()
        log_prob_z = log_prob_zc + log_prob_zs
        return log_prob_x_z, self.y_mult * log_prob_y_zc, self.q_mult * log_prob_z

    def inference(self, x):
        batch_size = len(x)
        z_param = nn.Parameter(torch.zeros(batch_size, 2 * self.z_size, device=self.device))
        nn.init.xavier_normal_(z_param)
        optim = Adam([z_param], lr=self.lr_inference)
        optim_loss = torch.inf
        optim_log_prob_x_z = optim_log_prob_y_zc = optim_log_prob_z = optim_z = None
        for _ in range(self.n_steps):
            optim.zero_grad()
            log_prob_x_z, log_prob_y_zc, log_prob_z = self.inference_loss(x, z_param)
            loss = -log_prob_x_z - log_prob_y_zc - log_prob_z
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
        with torch.set_grad_enabled(True):
            x, y = batch
            y_pred, log_prob_x_z, log_prob_y_zc, log_prob_z, loss = self.inference(x)
            self.log('test_log_prob_x_z', log_prob_x_z, on_step=False, on_epoch=True)
            self.log('test_log_prob_y_zc', log_prob_y_zc, on_step=False, on_epoch=True)
            self.log('test_log_prob_z', log_prob_z, on_step=False, on_epoch=True)
            self.log('test_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.acc.update(y_pred_class, y.long())

    def on_test_epoch_end(self):
        self.log('test_acc', self.acc.compute())

    def configure_grad(self):
        if self.stage == 'train':
            for params in self.train_params:
                params.requires_grad = True
            for params in self.train_q_params:
                params.requires_grad = False
        elif self.stage == 'train_q':
            for params in self.train_params:
                params.requires_grad = False
            for params in self.train_q_params:
                params.requires_grad = True
        else:
            for params in self.train_params:
                params.requires_grad = False
            for params in self.train_q_params:
                params.requires_grad = False

    def configure_optimizers(self):
        if self.stage == 'train':
            return Adam(self.train_params, lr=self.lr, weight_decay=self.weight_decay)
        elif self.stage == 'train_q':
            return Adam(self.train_q_params, lr=self.lr, weight_decay=self.weight_decay)