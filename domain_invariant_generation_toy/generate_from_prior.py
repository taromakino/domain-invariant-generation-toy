import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import MAKE_DATA, PLOT, IMAGE_SHAPE
from models.vae import VAE
from utils.file import load_file


def sample_prior(rng, vae, y_train, e_idx_train):
    idx = rng.choice(len(y_train), 1)
    prior_dist = vae.prior_dist(y_train[idx], e_idx_train[idx])
    z_sample = prior_dist.sample()
    return torch.chunk(z_sample, 2, dim=1)


def main(args):
    rng = np.random.RandomState(args.seed)
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, _, _ = MAKE_DATA[existing_args.dataset](existing_args.train_ratio, existing_args.batch_size)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    x_train, y_train, e_train = data_train.dataset[:]
    x_train, y_train, e_train = x_train.to(vae.device), y_train.to(vae.device), e_train.to(vae.device)
    y_idx_train = y_train.int()[:, 0]
    e_idx_train = e_train.int()[:, 0]
    for example_idx in range(args.n_examples):
        x_seed, y_idx_seed, e_idx_seed = x_train[[example_idx]], y_idx_train[[example_idx]], e_idx_train[[example_idx]]
        posterior_dist_seed = vae.posterior_dist(x_seed, y_idx_seed, e_idx_seed)
        z_seed = posterior_dist_seed.loc
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        fig, axes = plt.subplots(2, args.n_cols, figsize=(2 * args.n_cols, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot = PLOT[existing_args.dataset]
        image_size = IMAGE_SHAPE[existing_args.dataset]
        plot(axes[0, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        x_pred = torch.sigmoid(vae.decoder(z_seed))
        plot(axes[0, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        for col_idx in range(2, args.n_cols):
            zc_sample, zs_sample = sample_prior(rng, vae, y_idx_train, e_idx_train)
            x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_sample, zs_seed))))
            x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_sample))))
            plot(axes[0, col_idx], x_pred_causal.reshape(image_size).detach().cpu().numpy())
            plot(axes[1, col_idx], x_pred_spurious.reshape(image_size).detach().cpu().numpy())
        fig_dpath = os.path.join(args.dpath, f'version_{args.seed}', 'fig', 'generate_from_prior')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())