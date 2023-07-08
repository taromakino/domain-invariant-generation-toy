import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class VAE(pl.LightningModule):
    def __init__(self, z_size, h_sizes, n_classes, n_envs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.z_size = z_size
        self.n_classes = n_classes
        self.n_envs = n_envs
        self.half_z_size = z_size // 2
        # q(z_c,z_s|x,y,e)
        self.image_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 7, 1, 0),
            nn.ReLU(),
            nn.Conv2d(128, 50, 1, 1, 0),
        )
        self.encoder_mu = MLP(50, h_sizes, n_classes * n_envs * z_size, nn.ReLU)
        self.encoder_cov = MLP(50, h_sizes, n_classes * n_envs * size_to_n_tril(z_size), nn.ReLU)
        # p(x|z_c, z_s)
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(z_size, 128, 1, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 7, 1, 0),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 2, 4, 2, 1)
        )
        # p(y|z_c)
        self.causal_classifier = MLP(self.half_z_size, h_sizes, 1, nn.ReLU)
        # p(z_c|e)
        self.prior_mu_causal = nn.Parameter(torch.zeros(n_envs, self.half_z_size))
        self.prior_cov_causal = nn.Parameter(torch.zeros(n_envs, size_to_n_tril(self.half_z_size)))
        nn.init.xavier_normal_(self.prior_mu_causal)
        nn.init.xavier_normal_(self.prior_cov_causal)
        # p(z_s|y,e)
        self.prior_mu_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, self.half_z_size))
        self.prior_cov_spurious = nn.Parameter(torch.zeros(n_classes, n_envs, size_to_n_tril(self.half_z_size)))
        nn.init.xavier_normal_(self.prior_mu_spurious)
        nn.init.xavier_normal_(self.prior_cov_spurious)

    def sample_z(self, mu, cov):
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(cov, epsilon).squeeze()

    def forward(self, x, y, e):
        batch_size = len(x)
        y_idx = y.squeeze().int()
        e_idx = e.squeeze().int()
        # z_c, z_s ~ q(z_c,z_s|x,y,e)
        image_embedding = self.image_encoder(x).flatten(start_dim=1)
        posterior_mu = self.encoder_mu(image_embedding)
        posterior_mu = posterior_mu.reshape(batch_size, self.n_classes, self.n_envs, self.z_size)
        posterior_mu = posterior_mu[torch.arange(batch_size), y_idx, e_idx, :]
        posterior_cov = self.encoder_cov(image_embedding)
        posterior_cov = posterior_cov.reshape(batch_size, self.n_classes, self.n_envs, size_to_n_tril(self.z_size))
        posterior_cov = arr_to_scale_tril(posterior_cov[torch.arange(batch_size), y_idx, e_idx, :])
        z = self.sample_z(posterior_mu, posterior_cov)
        # E_q(z_c,z_s|x,y,e)[log p(x|z_c,z_s)]
        x_pred = self.decoder(z[:, :, None, None]).flatten(start_dim=1)
        x = x.flatten(start_dim=1)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # E_q(z_c,z_s|x,y,e)[log p(y|z_c)]
        z_c, z_s = torch.chunk(z, 2, dim=1)
        y_pred = self.causal_classifier(z_c)
        log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y, reduction='none')
        # KL(q(z_c,z_s|x,u) || p(z_c|e)p(z_s|y,e))
        posterior_dist = D.MultivariateNormal(posterior_mu, scale_tril=posterior_cov)
        prior_mu_causal, prior_cov_causal = self.prior_causal_params(e_idx)
        prior_mu_spurious, prior_cov_spurious = self.prior_spurious_params(y_idx, e_idx)
        prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
        # Block diagonal covariance matrix
        prior_cov = torch.zeros(batch_size, self.z_size, self.z_size, device=self.device)
        prior_cov[:, :self.half_z_size, :self.half_z_size] = prior_cov_causal
        prior_cov[:, self.half_z_size:, self.half_z_size:] = prior_cov_spurious
        prior_dist = D.MultivariateNormal(prior_mu, prior_cov)
        kl = D.kl_divergence(posterior_dist, prior_dist)
        elbo = log_prob_x_z + log_prob_y_zc - kl
        return -elbo.mean()

    def prior_causal_params(self, e_idx):
        prior_mu_causal = self.prior_mu_causal[e_idx]
        prior_cov_tril_causal = arr_to_scale_tril(self.prior_cov_causal[e_idx])
        prior_cov_causal = torch.bmm(prior_cov_tril_causal, torch.transpose(prior_cov_tril_causal, 1, 2))
        return prior_mu_causal, prior_cov_causal

    def prior_spurious_params(self, y_idx, e_idx):
        prior_mu_spurious = self.prior_mu_spurious[y_idx, e_idx]
        prior_cov_tril_spurious = arr_to_scale_tril(self.prior_cov_spurious[y_idx, e_idx])
        prior_cov_spurious = torch.bmm(prior_cov_tril_spurious, torch.transpose(prior_cov_tril_spurious, 1, 2))
        return prior_mu_spurious, prior_cov_spurious

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


class SpuriousClassifier(pl.LightningModule):
    def __init__(self, z_size, h_sizes, n_envs, lr):
        super().__init__()
        self.save_hyperparameters()
        self.n_envs = n_envs
        self.lr = lr
        # p(y|z_s, e)
        self.net = MLP(z_size // 2, h_sizes, n_envs, nn.ReLU)

    def forward(self, z_s, y, e):
        batch_size = len(z_s)
        e_idx = e.squeeze().int()
        y_pred = self.net(z_s).reshape(batch_size, self.n_envs, 1)
        y_pred = y_pred[torch.arange(batch_size), e_idx]
        return F.binary_cross_entropy_with_logits(y_pred, y)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)