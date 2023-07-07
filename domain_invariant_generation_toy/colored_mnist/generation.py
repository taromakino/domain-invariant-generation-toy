import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from colored_mnist.data import make_data
from model import Model, SpuriousClassifier
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'), 
        map_location='cpu')
    p_y_zs = SpuriousClassifier.load_from_checkpoint(os.path.join(args.dpath, 'spurious_classifier',
        f'version_{args.seed}', 'checkpoints', 'best.ckpt'), map_location='cpu').p_y_zs
    x, y, e = data_train.dataset[:]
    z = model.encoder_mu(x, y, e)
    z_c, z_s = torch.chunk(z, 2, dim=1)
    x_seed, y_seed, e_seed = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    x_seed, y_seed = x_seed[None], y_seed[None]
    zc_seed = z_c[args.example_idx][None]
    zs_seed = z_s[args.example_idx][None]
    fig, axes = plt.subplots(2, args.n_cols)
    plot_red_green_image(axes[0, 0], x_seed.reshape((2, 14, 14)).detach().numpy())
    plot_red_green_image(axes[1, 0], x_seed.reshape((2, 14, 14)).detach().numpy())
    zc_perturb = zc_seed.clone().requires_grad_(True)
    zs_perturb = zs_seed.clone().requires_grad_(True)
    for col_idx in range(1, args.n_cols):
        for _ in range(args.n_steps_per_col):
            y_pred_causal = model.p_y_zc(zc_perturb)
            loss_causal = F.binary_cross_entropy_with_logits(y_pred_causal, 1 - y_seed)
            grad_causal = torch.autograd.grad(loss_causal, zc_perturb)[0]
            zc_perturb = zc_perturb - args.eta * grad_causal
            y_pred_spurious = p_y_zs(zs_perturb)
            loss_spurious = F.binary_cross_entropy_with_logits(y_pred_spurious, 1 - y_seed)
            grad_spurious = torch.autograd.grad(loss_spurious, zs_perturb)[0]
            zs_perturb = zs_perturb - args.eta * grad_spurious
        x_pred_causal = model.decoder(torch.hstack((zc_perturb, zs_seed)))
        x_pred_spurious = model.decoder(torch.hstack((zc_seed, zs_perturb)))
        plot_red_green_image(axes[0, col_idx], x_pred_causal.reshape((2, 14, 14)).detach().numpy())
        plot_red_green_image(axes[1, col_idx], x_pred_spurious.reshape((2, 14, 14)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=5)
    parser.add_argument('--n_steps_per_col', type=int, default=100)
    parser.add_argument('--eta', type=float, default=1)
    main(parser.parse_args())