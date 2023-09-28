import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import MAKE_DATA, PLOT, IMAGE_SHAPE
from models.vae import VAE
from utils.enums import Task
from utils.file import load_file


def main(args):
    task_dpath = os.path.join(args.dpath, Task.Q_Z.value)
    existing_args = load_file(os.path.join(task_dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, _, _ = MAKE_DATA[existing_args.dataset](existing_args.train_ratio, existing_args.batch_size)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    q = model.q()
    x, y, e, _, _ = data_train.dataset[:]
    for example_idx in range(args.n_examples):
        x_seed, y_seed, e_seed = x[[example_idx]], y[[example_idx]], e[[example_idx]]
        x_seed, y_seed, e_seed = x_seed.to(model.device), y_seed.to(model.device), e_seed.to(model.device)
        posterior_dist_seed = model.encoder(x_seed)
        z_seed = posterior_dist_seed.loc
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        fig, axes = plt.subplots(2, args.n_cols, figsize=(2 * args.n_cols, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot = PLOT[existing_args.dataset]
        image_size = IMAGE_SHAPE[existing_args.dataset]
        plot(axes[0, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        x_pred = torch.sigmoid(model.decoder.mlp(z_seed))
        plot(axes[0, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        for col_idx in range(2, args.n_cols):
            z_sample = q.sample()[None]
            zc_sample, zs_sample = torch.chunk(z_sample, 2, dim=1)
            x_pred_causal = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_sample, zs_seed))))
            x_pred_spurious = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_seed, zs_sample))))
            plot(axes[0, col_idx], x_pred_causal.reshape(image_size).detach().cpu().numpy())
            plot(axes[1, col_idx], x_pred_spurious.reshape(image_size).detach().cpu().numpy())
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'generate_sample_q')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))
        plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())