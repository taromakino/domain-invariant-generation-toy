import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from data import MAKE_DATA, PLOT, IMAGE_SHAPE
from torch.optim import Adam
from utils.file import load_file
from models.vae import VAE


def log_prob_yzc(y, z_c, vae, q_causal):
    y_pred = vae.classifier(z_c)
    log_prob_y_zc = -F.binary_cross_entropy_with_logits(y_pred, y)
    log_prob_zc = q_causal.log_prob(z_c).mean()
    return log_prob_y_zc, log_prob_zc


def main(args):
    existing_args = load_file(os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, _, _ = MAKE_DATA[existing_args.dataset](existing_args.train_ratio, existing_args.batch_size)
    vae = VAE.load_from_checkpoint(os.path.join(args.dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'),
        stage='test')
    q_causal = vae.q_causal()
    x_train, y_train, e_train = data_train.dataset[:]
    x_train, y_train, e_train = x_train.to(vae.device), y_train.to(vae.device), e_train.to(vae.device)
    posterior_dist = vae.encoder(x_train, y_train, e_train)
    z = posterior_dist.loc
    for example_idx in range(args.n_examples):
        x_seed, y_seed, z_seed = x_train[[example_idx]], y_train[[example_idx]], z[[example_idx]]
        fig, axes = plt.subplots(1, args.n_cols, figsize=(2 * args.n_cols, 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot = PLOT[existing_args.dataset]
        image_size = IMAGE_SHAPE[existing_args.dataset]
        plot(axes[0], x_seed.reshape(image_size).detach().cpu().numpy())
        x_pred = torch.sigmoid(vae.decoder.mlp(z_seed))
        plot(axes[1], x_pred.reshape(image_size).detach().cpu().numpy())
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        zc_param = nn.Parameter(zc_seed)
        optimizer = Adam([zc_param], lr=args.lr_generate)
        for col_idx in range(2, args.n_cols):
            for _ in range(args.n_steps_per_col):
                optimizer.zero_grad()
                log_prob_y_zc, log_prob_zc = log_prob_yzc(1 - y_seed, zc_param, vae, q_causal)
                loss = args.y_mult * -log_prob_y_zc - log_prob_zc
                loss.backward()
                optimizer.step()
            x_perturb = torch.sigmoid(vae.decoder.mlp(torch.hstack((zc_param, zs_seed))))
            plot(axes[col_idx], x_perturb.reshape(image_size).detach().cpu().numpy())
        fig_dpath = os.path.join(args.dpath, f'version_{args.seed}', 'fig', 'generate')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--y_mult', type=float, default=1000)
    parser.add_argument('--lr_generate', type=float, default=0.01)
    parser.add_argument('--n_steps_per_col', type=int, default=1000)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())