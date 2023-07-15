import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.distributions as D
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE
from utils.file import load_file
from utils.plot import plot_red_green_image


def approx_prior(z):
    z_mu = z.mean(dim=0)
    z_cov = torch.cov(torch.swapaxes(z, 0, 1))
    if len(z_cov.shape) == 0:
        z_cov = z_cov.view(1, 1)
    return D.MultivariateNormal(z_mu, z_cov)


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    e_train, digits_train, y_train, colors_train, x_train = data_train.dataset[:]
    y_idx_train = y_train.int()[:, 0]
    e_idx_train = e_train.int()[:, 0]
    posterior_dist = vae.posterior_dist(x_train, y_idx_train, e_idx_train)
    z_train = posterior_dist.loc.detach()
    zc_train, zs_train = torch.chunk(z_train, 2, dim=1)
    zc_prior = approx_prior(zc_train)
    zs_prior = approx_prior(zs_train)
    idxs = ((digits_train == y_train) & (y_train == colors_train) & (e_train == 0)).squeeze()
    e_train, digits_train, y_train, colors_train, x_train, z_train = e_train[idxs], digits_train[idxs], y_train[idxs], \
        colors_train[idxs], x_train[idxs], z_train[idxs]
    x_seed, y_seed, e_seed, z_seed = x_train[args.example_idx], y_train[args.example_idx], e_train[args.example_idx], \
        z_train[args.example_idx]
    x_seed, y_seed, e_seed, z_seed = x_seed[None], y_seed[None], e_seed[None], z_seed[None]
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
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
        zc_sample = zc_prior.sample()
        zs_sample = zs_prior.sample()
        zc_perturb = args.alpha * zc_sample + (1 - args.alpha) * zc_seed
        zs_perturb = args.alpha * zs_sample + (1 - args.alpha) * zs_seed
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_perturb))))
        plot_red_green_image(axes[0, col_idx], x_pred_causal.reshape((2, 28, 28)).detach().numpy())
        plot_red_green_image(axes[1, col_idx], x_pred_spurious.reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=1)
    main(parser.parse_args())