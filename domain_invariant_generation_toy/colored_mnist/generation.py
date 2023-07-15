import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torch.distributions as D
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE
from torch.optim import SGD
from utils.file import load_file
from utils.plot import plot_red_green_image


def approx_prior(z):
    z_mu = z.mean(dim=0)
    z_cov = torch.cov(torch.swapaxes(z, 0, 1))
    if len(z_cov.shape) == 0:
        z_cov = z_cov.view(1, 1)
    return D.MultivariateNormal(z_mu, z_cov)


def loss_causal(vae, zc_prior, x, y, zc, zs):
    x_pred = vae.decoder(torch.hstack((zc, zs)))
    log_prob_x_z = -F.binary_cross_entropy_with_logits(x_pred, x, reduction='none').sum(dim=1).mean()
    y_pred = vae.causal_classifier(zc)
    log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
    log_prob_zc = zc_prior.log_prob(zc).mean()
    return -log_prob_x_z - log_prob_y_zc - log_prob_zc


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
    idxs = ((digits_train == y_train) & (y_train == colors_train) & (e_train == 0)).squeeze()
    e_train, digits_train, y_train, colors_train, x_train, z_train = e_train[idxs], digits_train[idxs], y_train[idxs], \
        colors_train[idxs], x_train[idxs], z_train[idxs]
    x_seed, y_seed, e_seed, z_seed = x_train[args.example_idx], y_train[args.example_idx], \
        e_train[args.example_idx], z_train[args.example_idx]
    x_seed, y_seed, e_seed, z_seed = x_seed[None], y_seed[None], e_seed[None], z_seed[None]
    fig, axes = plt.subplots(1, args.n_cols)
    fig.suptitle(f'y={y_seed.item()}, e={e_seed.item()}')
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    x_pred = torch.sigmoid(vae.decoder(z_seed))
    plot_red_green_image(axes[0], x_seed.reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1], x_pred.reshape((2, 28, 28)).detach().numpy())
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    zc_perturb = zc_seed.clone().requires_grad_(True)
    zc_optim = SGD([zc_perturb], lr=args.lr)
    for col_idx in range(2, args.n_cols):
        for _ in range(args.n_steps_per_col):
            zc_optim.zero_grad()
            loss_causal_step = loss_causal(vae, zc_prior, x_seed, 1 - y_seed, zc_perturb, zs_seed)
            loss_causal_step.backward()
            zc_optim.step()
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        plot_red_green_image(axes[col_idx], x_pred_causal.reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--n_steps_per_col', type=int, default=1000)
    parser.add_argument('--lr', type=float, default=1e-3)
    main(parser.parse_args())