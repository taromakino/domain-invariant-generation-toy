import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from utils.nn_utils import arr_to_scale_tril, size_to_n_tril


class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc_1 = nn.Linear(10, 32 * 4 * 4)
        self.deconv_1 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 0)
        self.deconv_2 = nn.ConvTranspose2d(32, 32, 3, 2, 1, 1)
        self.deconv_3 = nn.ConvTranspose2d(32, 2, 3, 2, 1, 1)

    def forward(self, z):
        out = F.relu(self.fc_1(z))
        out = out.view(-1, 32, 4, 4)
        out = F.relu(self.deconv_1(out))
        out = F.relu(self.deconv_2(out))
        out = F.relu(self.deconv_3(out))
        out = out + 0.1 * torch.randn_like(out)
        return torch.sigmoid(out)


class Model(pl.LightningModule):
    def __init__(self, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        # q(z|u,x)
        self.x_encoder = nn.Sequential(
            nn.Conv2d(2, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, 2, 1),
            nn.ReLU(),
            nn.Flatten()
        )
        self.encoder_embed = nn.Sequential(
            nn.Linear(516, 100),
            nn.ReLU()
        )
        self.encoder_mu = nn.Linear(100, 10)
        self.encoder_var = nn.Linear(100, 10)
        # p(x|z)
        self.decoder = Decoder()
        # p(z|u)
        self.prior_embed = nn.Sequential(
            nn.Linear(4, 50),
            nn.ReLU()
        )
        self.prior_mu = nn.Linear(50, 10)
        self.prior_cov = nn.Linear(50, size_to_n_tril(10))

    def sample_z(self, mu, var):
        sd = var.sqrt()
        epsilon = torch.randn_like(sd)
        return mu + sd * epsilon

    def forward(self, x, y, e):
        u = torch.cat((y, e), dim=1)
        # z ~ q(z|u,x)
        x_embed = self.x_encoder(x)
        embed_z_ux = self.encoder_embed(torch.cat((x_embed, u), dim=1))
        mu_z_ux = self.encoder_mu(embed_z_ux)
        var_z_ux = F.softplus(self.encoder_var(embed_z_ux))
        z = self.sample_z(mu_z_ux, var_z_ux)
        # E_q(z|u,x)[log p(x|z)]
        x_pred = self.decoder(z)
        log_prob_x_uz = -F.binary_cross_entropy(x_pred.flatten(start_dim=1), x.flatten(start_dim=1), reduction='none').sum(dim=1)
        # KL(q(z|u,x) || p(z|u))
        dist_z_ux = D.MultivariateNormal(mu_z_ux, torch.diag_embed(var_z_ux))
        embed_z_u = self.prior_embed(u)
        mu_z_u = self.prior_mu(embed_z_u)
        cov_z_u = arr_to_scale_tril(self.prior_cov(embed_z_u))
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