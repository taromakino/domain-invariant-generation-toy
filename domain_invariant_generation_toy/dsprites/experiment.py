import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from dsprites.data import make_data
from dsprites.model import VAE
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val = make_data(args.train_ratio, args.batch_size, args.n_workers)
    _, y, e = data_train.dataset[:]
    n_envs = int(e.max() + 1)
    vae = VAE(64 * 64, args.z_size, args.h_sizes, n_envs, args.prior_reg_mult, args.lr)
    vae_trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    vae_trainer.fit(vae, data_train, data_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z_size', type=int, default=20)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--prior_reg_mult', type=float, default=0)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=10)
    args = parser.parse_args()
    main(args)
