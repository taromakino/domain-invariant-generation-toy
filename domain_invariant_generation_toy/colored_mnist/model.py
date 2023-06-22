import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from ..utils.nn_utils import MLP
from ..utils.stats import arr_to_scale_tril, size_to_n_tril


class GaussianMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_class):
        super().__init__()
        self.mu_net = MLP(input_dim, hidden_dims, output_dim, activation_class)
        self.logvar_net = MLP(input_dim, hidden_dims, output_dim, activation_class)

    def forward(self, *args):
        return self.mu_net(*args), F.softplus(self.logvar_net(*args))


class Model(pl.LightningModule):
    def __init__(self, x_size, y_size, e_size, z_size, h_sizes, beta, lr):
        super().__init__()
        self.save_hyperparameters()
        self.beta = beta
        self.lr = lr
        u_size = e_size + y_size
        self.q_z_ux = GaussianMLP(u_size + x_size, h_sizes, z_size, nn.ReLU)
        self.p_x_uz = MLP(u_size + z_size, h_sizes, x_size, nn.ReLU)
        self.p_z_u_mu = MLP(u_size, h_sizes, z_size, nn.ReLU)
        self.p_z_u_cov = MLP(u_size, h_sizes, size_to_n_tril(z_size), nn.ReLU)
        self.p_y_x = MLP(x_size, h_sizes, y_size, nn.ReLU)


    def sample_z(self, mu, var):
        sd = var.sqrt()
        epsilon = torch.randn_like(sd)
        return mu + sd * epsilon


    def forward(self, x, y, e):
        u = torch.cat((e, y), dim=1)
        # z ~ q(z|u,x)
        mu_z_ux, var_z_ux = self.q_z_ux(u, x)
        z = self.sample_z(mu_z_ux, var_z_ux)
        # E_q(z|u,x)[log p(x|u,z)]
        x_pred = self.p_x_uz(u, z)
        log_prob_x_uz = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
        # KL(q(z|u,x) || p(z|u))
        dist_z_ux = D.MultivariateNormal(mu_z_ux, torch.diag_embed(var_z_ux))
        mu_z_u = self.p_z_u_mu(u)
        cov_z_u = arr_to_scale_tril(self.p_z_u_cov(u))
        dist_z_u = D.MultivariateNormal(mu_z_u, scale_tril=cov_z_u)
        kl = D.kl_divergence(dist_z_ux, dist_z_u)
        elbo = log_prob_x_uz - kl
        # BCE(y_pred, y)
        y_pred = self.p_y_x(x)
        nll_y_x = F.binary_cross_entropy_with_logits(y_pred, y)
        return -elbo.mean() + self.beta * nll_y_x


    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)