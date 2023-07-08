import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from colored_mnist.data import make_data
from colored_mnist.model import VAE, SpuriousClassifier
from utils.file import save_file
from utils.nn_utils import make_dataloader, make_trainer


def make_spurious_data(data, model, n_classes, n_envs, batch_size, n_workers, is_train):
    x, y, e = data.dataset[:]
    x, y, e = x.to(model.device), y.to(model.device), e.to(model.device)
    y_idx = y.squeeze().int()
    e_idx = e.squeeze().int()
    n_examples = len(x)
    image_embedding = model.image_encoder(x).flatten(start_dim=1)
    z = model.encoder_mu(image_embedding)
    z = z.reshape(n_examples, n_classes, n_envs, model.z_size)
    z = z[torch.arange(n_examples), y_idx, e_idx, :]
    z_c, z_s = torch.chunk(z, 2, dim=1)
    return make_dataloader((z_s.detach().cpu(), y.cpu(), e.cpu()), batch_size, n_workers, is_train)


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val = make_data(args.train_ratio, args.batch_size, args.n_workers)
    _, y, e = data_train.dataset[:]
    n_classes = int(y.max() + 1)
    n_envs = int(e.max() + 1)
    model = VAE(args.z_size, args.h_sizes, n_classes, n_envs, args.lr)
    model_trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    model_trainer.fit(model, data_train, data_val)
    model = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    zs_y_data_train = make_spurious_data(data_train, model, n_classes, n_envs, args.batch_size, args.n_workers, True)
    zs_y_data_val = make_spurious_data(data_val, model, n_classes, n_envs, args.batch_size, args.n_workers, False)
    spurious_classifier = SpuriousClassifier(args.z_size, args.h_sizes, n_envs, args.lr)
    spurious_classifier_trainer = make_trainer(os.path.join(args.dpath, 'spurious_classifier'), args.seed,
        args.n_epochs, args.early_stop_ratio)
    spurious_classifier_trainer.fit(spurious_classifier, zs_y_data_train, zs_y_data_val)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--z_size', type=int, default=20)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=10)
    args = parser.parse_args()
    assert args.z_size % 2 == 0
    main(args)
