import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from argparse import ArgumentParser
from colored_mnist.data import make_data
from model import Model
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'), map_location='cpu')
    x, y, e = data_train.dataset[:]
    z = model.encoder_mu(x, y, e)
    z_c, z_s = torch.chunk(z, 2, dim=1)
    x_example, y_example, e_example = x[args.example_idx], y[args.example_idx], e[args.example_idx]
    x_example, y_example, e_example = x_example[None], y_example[None], e_example[None]
    x_example, y_example, e_example = x_example.to(model.device), y_example.to(model.device), e_example.to(model.device)
    zc_example = z_c[args.example_idx][None].requires_grad_(True)
    zs_example = z_s[args.example_idx][None]
    fig, axes = plt.subplots(1, args.n_cols)
    plot_red_green_image(axes[0], x_example.reshape((2, 14, 14)).detach().numpy())
    for col_idx in range(1, args.n_cols):
        zc_example.grad = None
        y_pred = model.p_y_zc(zc_example)
        loss = F.binary_cross_entropy_with_logits(y_pred, 1 - y_example)
        grad = torch.autograd.grad(loss, zc_example)[0]
        zc_example = zc_example - args.alpha * grad
        x_pred = model.decoder(torch.hstack((zc_example, zs_example)))
        plot_red_green_image(axes[col_idx], x_pred.reshape((2, 14, 14)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--example_idx', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=100)
    parser.add_argument('--n_cols', type=int, default=5)
    main(parser.parse_args())