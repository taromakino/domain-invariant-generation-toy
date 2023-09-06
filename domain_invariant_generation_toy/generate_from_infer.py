import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import PLOT, IMAGE_SHAPE
from models.model import Model
from utils.enums import Task
from utils.file import load_file


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.TRAIN_Q.value)
    existing_args = load_file(os.path.join(task_dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    model = Model.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    if args.stage == 'train':
        z, y, c, s, x = torch.load(os.path.join(args.dpath, Task.INFER_Z_TRAIN.value, f'version_{args.seed}', 'infer.pt'))
    elif args.stage == 'val':
        z, y, c, s, x = torch.load(os.path.join(args.dpath, Task.INFER_Z_VAL.value, f'version_{args.seed}', 'infer.pt'))
    else:
        z, y, c, s, x = torch.load(os.path.join(args.dpath, Task.INFER_Z_TEST.value, f'version_{args.seed}', 'infer.pt'))
    for example_idx in range(args.n_examples):
        x_seed, z_seed = x[[example_idx]], z[[example_idx]].to(model.device)
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
            idx = rng.choice(args.n_examples, 1)
            z_sample = z[[idx]]
            zc_sample, zs_sample = torch.chunk(z_sample, 2, dim=1)
            x_pred_causal = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_sample, zs_seed))))
            x_pred_spurious = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_seed, zs_sample))))
            plot(axes[0, col_idx], x_pred_causal.reshape(image_size).detach().cpu().numpy())
            plot(axes[1, col_idx], x_pred_spurious.reshape(image_size).detach().cpu().numpy())
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'generate_from_inferred', args.stage)
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--stage', type=str, choices=['train', 'val', 'test'], required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--n_cols', type=int, default=10)
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())