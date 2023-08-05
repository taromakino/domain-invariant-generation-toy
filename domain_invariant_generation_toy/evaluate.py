import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from data import MAKE_DATA
from vae import VAE
from torch.optim import Adam
from utils.file import load_file


def multivariate_normal(z):
    z_mu = z.mean(dim=0)
    z_cov = torch.cov(torch.swapaxes(z, 0, 1))
    return D.MultivariateNormal(z_mu, z_cov)


def loss(vae, q_zc, q_zs, x, z):
    x_pred = vae.decoder(z)
    log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1)
    z_c, z_s = torch.chunk(z, 2, dim=1)
    prob_y1_zc = torch.sigmoid(vae.causal_classifier(z_c))
    prob_y0_zc = 1 - prob_y1_zc
    prob_y_zc = torch.hstack((prob_y0_zc, prob_y1_zc))
    log_prob_y_zc = torch.log(prob_y_zc.max(dim=1).values)
    log_prob_zc = q_zc.log_prob(z_c)
    log_prob_zs = q_zs.log_prob(z_s)
    loss = (-log_prob_x_z - vae.classifier_mult * log_prob_y_zc - log_prob_zc - log_prob_zs).mean()
    y_pred = prob_y_zc.argmax(dim=1)
    return loss, y_pred


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, _, data_test = MAKE_DATA[existing_args.dataset](existing_args.train_ratio, existing_args.batch_size)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    vae.freeze()
    x_train, y_train, e_train = data_train.dataset[:]
    x_test, y_test = data_test.dataset[:]
    y_idx_train = y_train.int()[:, 0]
    e_idx_train = e_train.int()[:, 0]
    zc_train, zs_train = torch.chunk(vae.posterior_dist(x_train, y_idx_train, e_idx_train).loc, 2, dim=1)
    q_zc = multivariate_normal(zc_train)
    q_zs = multivariate_normal(zs_train)
    y_pred = []
    for x_batch, _ in data_test:
        batch_size = len(x_batch)
        z_batch = nn.Parameter(torch.zeros(batch_size, 2 * existing_args.z_size))
        nn.init.xavier_normal_(z_batch)
        optim = Adam([z_batch], lr=args.lr)
        optim_loss = np.inf
        optim_y_pred = None
        for _ in range(args.n_steps):
            optim.zero_grad()
            loss_batch, y_pred_batch = loss(vae, q_zc, q_zs, x_batch, z_batch)
            loss_batch.backward()
            optim.step()
            if loss_batch < optim_loss:
                optim_loss = loss_batch
                optim_y_pred = y_pred_batch.clone()
        y_pred.append(optim_y_pred)
    y_pred = torch.cat(y_pred, dim=0)
    acc_test = (y_pred == y_test.squeeze().int()).float().mean()
    print(f'acc_test={acc_test.item():.3f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.01)
    main(parser.parse_args())