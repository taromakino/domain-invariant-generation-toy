import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from dsprites.data import make_data
from dsprites.model import VAE
from torch.optim import Adam
from utils.file import load_file
from utils.plot import plot_grayscale_image
from utils.stats import multivariate_normal


def log_prob_yzc(vae, p_zc, y, zc):
    y_pred = vae.causal_classifier(zc)
    log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
    return log_prob_y_zc + p_zc.log_prob(zc).mean()


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    vae.freeze()
    x, y, e = data_train.dataset[:]
    e_idx = e.int()[:, 0]
    posterior_dist = vae.posterior_dist(x, y, e_idx)
    z = posterior_dist.loc
    zc, zs = torch.chunk(z, 2, dim=1)
    p_zc = multivariate_normal(zc)
    x_seed, y_seed, z_seed = x[[args.example_idx]], y[[args.example_idx]], z[[args.example_idx]]
    fig, axes = plt.subplots(1, args.n_cols)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot_grayscale_image(axes[0], x_seed.reshape((64, 64)).detach().numpy())
    x_pred = torch.sigmoid(vae.decoder(z_seed))
    plot_grayscale_image(axes[1], x_pred.reshape((64, 64)).detach().numpy())
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    zc_perturb = zc_seed.clone().requires_grad_()
    optimizer = Adam([zc_perturb], lr=args.lr)
    for col_idx in range(2, args.n_cols):
        for _ in range(args.n_steps_per_col):
            optimizer.zero_grad()
            loss = -log_prob_yzc(vae, p_zc, torch.ones(1, 1), zc_perturb)
            loss.backward()
            optimizer.step()
        print(loss)
        x_perturb = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        plot_grayscale_image(axes[col_idx], x_perturb.reshape((64, 64)).detach().numpy())
    fig_dpath = os.path.join(args.dpath, f'version_{args.seed}', 'fig', 'generate')
    os.makedirs(fig_dpath, exist_ok=True)
    plt.savefig(os.path.join(fig_dpath, f'{args.example_idx}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_steps_per_col', type=int, default=5000)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    main(parser.parse_args())