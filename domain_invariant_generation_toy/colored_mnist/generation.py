import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE, CausalPredictor, SpuriousPredictor
from torch.optim import Adam
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    causal_predictor = CausalPredictor.load_from_checkpoint(os.path.join(args.dpath, 'causal_predictor',
        f'version_{args.seed}', 'checkpoints', 'best.ckpt'), map_location='cpu')
    spurious_predictor = SpuriousPredictor.load_from_checkpoint(os.path.join(args.dpath, 'spurious_predictor',
        f'version_{args.seed}', 'checkpoints', 'best.ckpt'), map_location='cpu')
    vae.eval()
    causal_predictor.eval()
    spurious_predictor.eval()
    e, digits, y, colors, x = data_train.dataset[:]
    idxs = (digits == y == colors) and (e == 0)
    e, digits, y, colors, x = e[idxs], digits[idxs], y[idxs], colors[idxs], x[idxs]
    x_seed, y_seed, e_seed = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    x_seed, y_seed, e_seed = x_seed[None], y_seed[None], e_seed[None]
    y_idx_seed = y_seed.squeeze().int()
    e_idx_seed = e_seed.squeeze().int()
    posterior_dist = vae.posterior_dist(x_seed, y_idx_seed, e_idx_seed)
    z_seed = posterior_dist.loc.detach()
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
    zc_perturb = zc_seed.clone().requires_grad_(True)
    zs_perturb = zs_seed.clone().requires_grad_(True)
    zc_optim = Adam([zc_perturb], lr=args.lr)
    zs_optim = Adam([zs_perturb], lr=args.lr)
    for col_idx in range(2, args.n_cols):
        for _ in range(args.n_steps_per_col):
            zc_optim.zero_grad()
            loss_causal = causal_predictor(zc_perturb, 1 - y_seed)
            loss_causal.backward()
            zc_optim.step()
            zs_optim.zero_grad()
            loss_spurious = spurious_predictor(zs_perturb, 1 - y_seed, e_seed)
            loss_spurious.backward()
            zs_optim.step()
        x_pred_causal = torch.sigmoid(vae.decoder(torch.hstack((zc_perturb, zs_seed))))
        x_pred_spurious = torch.sigmoid(vae.decoder(torch.hstack((zc_seed, zs_perturb))))
        plot_red_green_image(axes[0, col_idx], x_pred_causal.reshape((2, 28, 28)).detach().numpy())
        plot_red_green_image(axes[1, col_idx], x_pred_spurious.reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--n_steps_per_col', type=int, default=5000)
    parser.add_argument('--lr', type=float, default=0.5)
    main(parser.parse_args())