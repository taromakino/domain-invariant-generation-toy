import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from dsprites.data import make_data
from dsprites.model import VAE, CausalPredictor, SpuriousPredictor
from utils.file import save_file
from utils.nn_utils import make_dataloader, make_trainer


def make_classify_data(data, vae, batch_size, n_workers, is_train):
    x, y, e = data.dataset[:]
    x, y, e = x.to(vae.device), y.to(vae.device), e.to(vae.device)
    y_idx = y.squeeze().int()
    e_idx = e.squeeze().int()
    posterior_dist = vae.posterior_dist(x, y_idx, e_idx)
    z = posterior_dist.loc
    z_c, z_s = torch.chunk(z, 2, dim=1)
    causal_data = make_dataloader((z_c.detach().cpu(), y.cpu()), batch_size, n_workers, is_train)
    spurious_data = make_dataloader((z_s.detach().cpu(), y.cpu(), e.cpu()), batch_size, n_workers, is_train)
    return causal_data, spurious_data


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val = make_data(args.train_ratio, args.batch_size, args.n_workers)
    _, y, e = data_train.dataset[:]
    n_envs = int(e.max() + 1)
    vae = VAE(64 * 64, args.z_size, args.h_sizes, n_envs, args.lr)
    vae_trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    vae_trainer.fit(vae, data_train, data_val)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    vae.eval()
    causal_data_train, spurious_data_train = make_classify_data(data_train, vae, args.batch_size, args.n_workers, True)
    causal_data_val, spurious_data_val = make_classify_data(data_val, vae, args.batch_size, args.n_workers, False)
    causal_predictor = CausalPredictor(args.z_size, args.h_sizes, args.lr)
    causal_predictor.net.load_state_dict(vae.causal_predictor.state_dict())
    spurious_predictor = SpuriousPredictor(args.z_size, args.h_sizes, n_envs, args.lr)
    causal_trainer = make_trainer(os.path.join(args.dpath, 'causal_predictor'), args.seed, args.n_epochs,
        args.early_stop_ratio)
    spurious_trainer = make_trainer(os.path.join(args.dpath, 'spurious_predictor'), args.seed, args.n_epochs,
        args.early_stop_ratio)
    causal_trainer.fit(causal_predictor, causal_data_train, causal_data_val)
    spurious_trainer.fit(spurious_predictor, spurious_data_train, spurious_data_val)


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
