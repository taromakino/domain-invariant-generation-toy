import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from dsprites.data import make_data
from dsprites.model import VAE
from utils.file import load_file
from utils.plot import plot_grayscale_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    x, y, e = data_train.dataset[:]
    x_seed, y_seed, e_seed = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    x_seed, y_seed, e_seed = x_seed[None], y_seed[None], e_seed[None]
    e_idx_seed = e_seed.int()[:, 0]
    posterior_dist = vae.posterior_dist(x_seed, y_seed, e_idx_seed)
    z_seed = posterior_dist.loc.detach()
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    prior_dist = vae.prior_dist(y_seed, e_idx_seed)
    fig, axes = plt.subplots(2, args.n_cols)
    fig.suptitle(f'y={y_seed.item():.3f}, e={e_seed.item()}')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot_grayscale_image(axes[0, 0], x_seed.reshape((64, 64)).detach().numpy())
    plot_grayscale_image(axes[1, 0], x_seed.reshape((64, 64)).detach().numpy())
    x_pred = torch.sigmoid(vae.decoder(z_seed))
    plot_grayscale_image(axes[0, 1], x_pred.reshape((64, 64)).detach().numpy())
    plot_grayscale_image(axes[1, 1], x_pred.reshape((64, 64)).detach().numpy())
    for col_idx in range(2, args.n_cols):
        z_sample = prior_dist.sample()
        zc_sample, zs_sample = torch.chunk(z_sample, 2, dim=1)
        zc_perturb = args.alpha * zc_sample + (1 - args.alpha) * zc_seed
        zs_perturb = args.alpha * zs_sample + (1 - args.alpha) * zs_seed
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_perturb))))
        plot_grayscale_image(axes[0, col_idx], x_pred_causal.reshape((64, 64)).detach().numpy())
        plot_grayscale_image(axes[1, col_idx], x_pred_spurious.reshape((64, 64)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    main(parser.parse_args())