import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import make_data
from model import Model
from utils.file import save_file
from utils.nn_utils import make_trainer


def main(args):
    save_file(args, os.path.join(args.dpath, 'args.pkl'))
    os.makedirs(args.dpath, exist_ok=True)
    pl.seed_everything(args.seed)
    data_train = make_data(args.batch_size, args.n_workers)
    model = Model(2 * 14 * 14, 1, 3, args.z_size, args.h_sizes, args.beta, args.lr)
    trainer = make_trainer(args.dpath, args.seed, args.n_epochs)
    trainer.fit(model, data_train)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z_size', type=int, default=10)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[256, 256])
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=1000)
    parser.add_argument('--n_workers', type=int, default=20)
    args = parser.parse_args()
    main(args)
