import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE, SpuriousClassifier
from torch.optim import Adam
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    causal_classifier = SpuriousClassifier.load_from_checkpoint(os.path.join(args.dpath, 'causal_classifier',
        f'version_{args.seed}', 'checkpoints', 'best.ckpt'), map_location='cpu')
    spurious_classifier = SpuriousClassifier.load_from_checkpoint(os.path.join(args.dpath, 'spurious_classifier',
        f'version_{args.seed}', 'checkpoints', 'best.ckpt'), map_location='cpu')
    vae.eval()
    causal_classifier.eval()
    spurious_classifier.eval()
    x, y, e = data_train.dataset[:]
    x_seed, y_seed, e_seed = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    # Generate in the environment where y and color are positively correlated
    assert torch.allclose(e_seed, torch.tensor([0.]))
    x_seed, y_seed, e_seed = x_seed[None], y_seed[None], e_seed[None]
    y_idx_seed = y_seed.squeeze().int()
    e_idx_seed = e_seed.squeeze().int()
    posterior_dist = vae.posterior_dist(x_seed, y_idx_seed, e_idx_seed)
    z_seed = posterior_dist.loc.detach()
    zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
    fig, axes = plt.subplots(2, args.n_cols)
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    plot_red_green_image(axes[0, 0], x_seed.reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1, 0], x_seed.reshape((2, 28, 28)).detach().numpy())
    zc_perturb = zc_seed.clone().requires_grad_(True)
    zs_perturb = zs_seed.clone().requires_grad_(True)
    zc_optim = Adam([zc_perturb], lr=args.lr)
    zs_optim = Adam([zs_perturb], lr=args.lr)
    for col_idx in range(1, args.n_cols):
        for _ in range(args.n_steps_per_col):
            zc_optim.zero_grad()
            loss_causal = causal_classifier(zs_perturb, 1 - y_seed, e_seed)
            loss_causal.backward()
            zc_optim.step()
            zs_optim.zero_grad()
            loss_spurious = spurious_classifier(zs_perturb, 1 - y_seed, e_seed)
            loss_spurious.backward()
            zs_optim.step()
        x_pred_causal = vae.decoder(torch.hstack((zc_perturb, zs_seed)))
        x_pred_spurious = vae.decoder(torch.hstack((zc_seed, zs_perturb)))
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
    parser.add_argument('--lr', type=float, default=1)
    main(parser.parse_args())