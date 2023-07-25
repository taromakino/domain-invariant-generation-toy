import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from dsprites.data import make_data
from dsprites.model import VAE
from utils.file import load_file
from utils.plot import plot_grayscale_image


def sample_prior(rng, vae, y_train, e_idx_train):
    idx = [rng.choice(len(y_train), 1).item()]
    prior_dist = vae.prior_dist(y_train[idx], e_idx_train[idx])
    z_sample = prior_dist.sample()
    return torch.chunk(z_sample, 2, dim=1)


def main(args):
    rng = np.random.RandomState(args.seed)
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    x_train, y_train, e_train = data_train.dataset[:]
    e_idx_train = e_train.int()[:, 0]
    x_seed, y_seed, e_seed = x_train[args.example_idx], y_train[args.example_idx], e_train[args.example_idx]
    x_seed, y_seed = x_seed[None], y_seed[None]
    posterior_dist_seed = vae.posterior_dist(x_seed, y_seed, e_seed.int())
    z_seed = posterior_dist_seed.loc
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    fig, axes = plt.subplots(2, args.n_cols)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot_grayscale_image(axes[0, 0], x_seed.reshape((64, 64)).detach().numpy())
    plot_grayscale_image(axes[1, 0], x_seed.reshape((64, 64)).detach().numpy())
    x_pred = torch.sigmoid(vae.decoder(z_seed))
    plot_grayscale_image(axes[0, 1], x_pred.reshape((64, 64)).detach().numpy())
    plot_grayscale_image(axes[1, 1], x_pred.reshape((64, 64)).detach().numpy())
    for col_idx in range(2, args.n_cols):
        zc_sample, zs_sample = sample_prior(rng, vae, y_train, e_idx_train)
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_sample, zs_seed))))
        x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_sample))))
        plot_grayscale_image(axes[0, col_idx], x_pred_causal.reshape((64, 64)).detach().numpy())
        plot_grayscale_image(axes[1, col_idx], x_pred_spurious.reshape((64, 64)).detach().numpy())
    fig_dpath = os.path.join(args.dpath, f'version_{args.seed}', 'fig', 'generate_from_prior')
    os.makedirs(fig_dpath, exist_ok=True)
    plt.savefig(os.path.join(fig_dpath, f'{args.example_idx}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--example_idx', type=int, default=0)
    main(parser.parse_args())