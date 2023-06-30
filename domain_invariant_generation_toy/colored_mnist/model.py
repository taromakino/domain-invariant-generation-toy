import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import MLP, arr_to_scale_tril, size_to_n_tril


class Model(pl.LightningModule):
    def __init__(self, x_size, y_size, e_size, z_size, h_sizes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        u_size = e_size + y_size
        # q(z|u,x)
        self.encoder_mu = MLP(u_size + x_size, h_sizes, z_size, nn.ReLU)
        self.encoder_var = MLP(u_size + x_size, h_sizes, z_size, nn.ReLU)
        # p(x|z)
        self.decoder = MLP(z_size, h_sizes, x_size, nn.ReLU)
        # p(z|u)
        self.prior_mu = MLP(u_size, h_sizes, z_size, nn.ReLU)
        self.prior_cov = MLP(u_size, h_sizes, size_to_n_tril(z_size), nn.ReLU)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        epsilon = torch.randn_like(sd)
        return mu + sd * epsilon


    def forward(self, x, y, e):
        u = torch.cat((y, e), dim=1)
        # z ~ q(z|u,x)
        mu_z_ux = self.encoder_mu(u, x)
        var_z_ux = F.softplus(self.encoder_var(u, x))
        z = self.sample_z(mu_z_ux, var_z_ux)
        # E_q(z|u,x)[log p(x|z)]
        x_pred = self.decoder(z)
        log_prob_x_uz = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # KL(q(z|u,x) || p(z|u))
        dist_z_ux = D.MultivariateNormal(mu_z_ux, torch.diag_embed(var_z_ux))
        mu_z_u = self.prior_mu(u)
        cov_z_u = arr_to_scale_tril(self.prior_cov(u))
        dist_z_u = D.MultivariateNormal(mu_z_u, scale_tril=cov_z_u)
        kl = D.kl_divergence(dist_z_ux, dist_z_u)
        elbo = log_prob_x_uz - kl
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