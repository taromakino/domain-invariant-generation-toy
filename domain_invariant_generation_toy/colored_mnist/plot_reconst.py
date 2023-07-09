import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE
from utils.file import load_file
from utils.plot import plot_red_green_image


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        map_location='cpu')
    vae.eval()
    x, y, e = data_train.dataset[:]
    x_seed, y_seed, e_seed = x[0], y[0], e[0]
    x_seed, y_seed, e_seed = x_seed[None], y_seed[None], e_seed[None]
    y_idx_seed = y_seed.squeeze().int()
    e_idx_seed = e_seed.squeeze().int()
    posterior_dist = vae.posterior_dist(x_seed, y_idx_seed, e_idx_seed)
    z_seed = posterior_dist.loc.detach()
    x_pred_seed = torch.sigmoid(vae.decoder(z_seed))
    fig, axes = plt.subplots(1, 2)
    plot_red_green_image(axes[0], x.reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1], x_pred_seed.reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())