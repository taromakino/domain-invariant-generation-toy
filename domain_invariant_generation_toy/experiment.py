import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import MAKE_DATA
from utils.file import save_file
from utils.nn_utils import make_trainer
from vae import VAE


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size, args.n_workers)
    x_train, y_train, e_train = data_train.dataset[:]
    n_classes = int(y_train.max() + 1)
    n_envs = int(e_train.max() + 1)
    vae = VAE(x_train.shape[1], args.z_size, args.h_sizes, n_classes, n_envs, args.classifier_mult,
        args.posterior_reg_mult, args.lr)
    vae_trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    vae_trainer.fit(vae, data_train, data_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z_size', type=int, default=50)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--classifier_mult', type=float, default=1)
    parser.add_argument('--posterior_reg_mult', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=10)
    args = parser.parse_args()
    main(args)
