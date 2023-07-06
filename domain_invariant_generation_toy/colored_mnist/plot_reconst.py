import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
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
    u = torch.cat((y, e), dim=1)
    z = model.encoder_mu(x, u)
    x_pred = torch.sigmoid(model.decoder(z))
    fig, axes = plt.subplots(1, 2)
    plot_red_green_image(axes[0], x[0].reshape((2, 14, 14)).detach().numpy())
    plot_red_green_image(axes[1], x_pred[0].reshape((2, 14, 14)).detach().numpy())
    plt.show(block=True)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    main(parser.parse_args())