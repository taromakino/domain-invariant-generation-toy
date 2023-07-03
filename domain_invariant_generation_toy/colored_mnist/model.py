import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class Model(pl.LightningModule):
    def __init__(self, x_size, y_size, e_size, z_size, h_sizes, beta, lr):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.lr = lr
        u_size = e_size + y_size
        # q(z|x,u)
        self.encoder_mu = MLP(x_size + u_size, h_sizes, z_size, nn.ReLU)
        self.encoder_cov = MLP(x_size + u_size, h_sizes, size_to_n_tril(z_size), nn.ReLU)
        # p(x|z)
        self.decoder = MLP(z_size, h_sizes, x_size, nn.ReLU)
        # p(z|u)
        self.prior_mu = MLP(u_size, h_sizes, z_size, nn.ReLU)
        self.prior_cov = MLP(u_size, h_sizes, size_to_n_tril(z_size), nn.ReLU)


    def sample_z(self, mu, cov):
        batch_size, z_size = mu.shape
        epsilon = torch.randn(batch_size, z_size, 1).to(self.device)
        return mu + torch.bmm(cov, epsilon).squeeze()


    def forward(self, x, y, e):
        u = torch.cat((y, e), dim=1)
        # z ~ q(z|x,u)
        mu_z_xu = self.encoder_mu(x, u)
        cov_z_xu = arr_to_scale_tril(self.encoder_cov(x, u))
        z = self.sample_z(mu_z_xu, cov_z_xu)
        # E_q(z|x,u)[log p(x|z)]
        x_pred = self.decoder(z)
        log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # KL(q(z|x,u) || p(z|u))
        dist_z_xu = D.MultivariateNormal(mu_z_xu, scale_tril=cov_z_xu)
        mu_z_u = self.prior_mu(u)
        cov_z_u = arr_to_scale_tril(self.prior_cov(u))
        dist_z_u = D.MultivariateNormal(mu_z_u, scale_tril=cov_z_u)
        kl = D.kl_divergence(dist_z_xu, dist_z_u)
        elbo = log_prob_x_z - self.beta * kl
        return -elbo.mean()


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