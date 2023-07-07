import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from colored_mnist.data import make_data
from model import Model, SpuriousClassifier
from utils.file import save_file
from utils.nn_utils import make_dataloader, make_trainer


def make_spurious_data(data, model, batch_size, n_workers, is_train):
    x, y, e = data.dataset[:]
    x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)
    z = model.encoder_mu(x, y, e)
    z_c, z_s = torch.chunk(z, 2, dim=1)
    return make_dataloader((z_s.detach().cpu(), y.cpu(), e.cpu()), batch_size, n_workers, is_train)


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val = make_data(args.train_ratio, args.batch_size, args.n_workers)
    model = Model(2 * 14 * 14, 1, 2, args.z_size, args.h_sizes, args.lr)
    model_trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    model_trainer.fit(model, data_train, data_val)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    zs_y_data_train = make_spurious_data(data_train, model, args.batch_size, args.n_workers, True)
    zs_y_data_val = make_spurious_data(data_val, model, args.batch_size, args.n_workers, False)
    spurious_classifier = SpuriousClassifier(1, 2, args.z_size, args.h_sizes, args.lr)
    spurious_classifier_trainer = make_trainer(os.path.join(args.dpath, 'spurious_classifier'), args.seed,
        args.n_epochs, args.early_stop_ratio)
    spurious_classifier_trainer.fit(spurious_classifier, zs_y_data_train, zs_y_data_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z_size', type=int, default=20)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=10)
    args = parser.parse_args()
    assert args.z_size % 2 == 0
    main(args)
