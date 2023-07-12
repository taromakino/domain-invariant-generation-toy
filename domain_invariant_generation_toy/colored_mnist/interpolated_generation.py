import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    rng = np.random.RandomState(args.seed)
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    e, digits, y, colors, x = data_train.dataset[:]
    idxs = ((digits == y) & (y == colors) & (e == 0)).squeeze()
    e, digits, y, colors, x = e[idxs], digits[idxs], y[idxs], colors[idxs], x[idxs]
    x_seed, y_seed, e_seed = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    x_seed, y_seed, e_seed = x_seed[None], y_seed[None], e_seed[None]
    y_idx_seed = y_seed.squeeze().int()
    e_idx_seed = e_seed.squeeze().int()
    posterior_dist = vae.posterior_dist(x_seed, y_idx_seed, e_idx_seed)
    z_seed = posterior_dist.loc.detach()
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    prior_mu_causal = vae.prior_mu_causal[e_idx_seed][None]
    prior_mu_spurious = vae.prior_mu_spurious[y_idx_seed, e_idx_seed][None]
    prior_mu = torch.hstack((prior_mu_causal, prior_mu_spurious))
    prior_cov = torch.eye(prior_mu.shape[1]).expand(1, prior_mu.shape[1], prior_mu.shape[1])
    prior_dist = D.MultivariateNormal(prior_mu, prior_cov)
    fig, axes = plt.subplots(2, args.n_cols)
    fig.suptitle(f'y={y_seed.item()}, e={e_seed.item()}')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot_red_green_image(axes[0, 0], x_seed.reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1, 0], x_seed.reshape((2, 28, 28)).detach().numpy())
    x_pred = torch.sigmoid(vae.decoder(z_seed))
    plot_red_green_image(axes[0, 1], x_pred.reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1, 1], x_pred.reshape((2, 28, 28)).detach().numpy())
    for col_idx in range(2, args.n_cols):
        alpha = rng.random()
        z_sample = prior_dist.sample()
        zc_sample, zs_sample = torch.chunk(z_sample, 2, dim=1)
        zc_perturb = alpha * zc_seed + (1 - alpha) * zc_sample
        zs_perturb = alpha * zs_seed + (1 - alpha) * zs_sample
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_perturb))))
        plot_red_green_image(axes[0, col_idx], x_pred_causal.reshape((2, 28, 28)).detach().numpy())
        plot_red_green_image(axes[1, col_idx], x_pred_spurious.reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=20)
    parser.add_argument('--example_idx', type=int, default=0)
    main(parser.parse_args())