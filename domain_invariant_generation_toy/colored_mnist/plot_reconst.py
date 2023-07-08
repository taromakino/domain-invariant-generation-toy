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
    n_classes = int(y.max() + 1)
    n_envs = int(e.max() + 1)
    y_idx = y.squeeze().int()
    e_idx = e.squeeze().int()
    n_examples = len(x)
    image_embedding = vae.image_encoder(x).flatten(start_dim=1)
    z = vae.encoder_mu(image_embedding)
    z = z.reshape(n_examples, n_classes, n_envs, vae.z_size)
    z = z[torch.arange(n_examples), y_idx, e_idx, :]
    x_pred = torch.sigmoid(vae.decoder(z[:, :, None, None]))
    fig, axes = plt.subplots(1, 2)
    plot_red_green_image(axes[0], x[0].reshape((2, 28, 28)).detach().numpy())
    plot_red_green_image(axes[1], x_pred[0].reshape((2, 28, 28)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())