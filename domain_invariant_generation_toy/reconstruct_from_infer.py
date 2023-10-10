import matplotlib.pyplot as plt
import numpy as np
import os
import pytorch_lightning as pl
import torch
from data import PLOT, IMAGE_SHAPE
from models.vae import VAE
from utils.enums import Task, EvalStage
from utils.nn_utils import make_dataloader


N_EXAMPLES = 10
N_COLS = 10


def sample_prior(rng, model, y, e):
    idx = rng.choice(len(y), 1)
    prior_dist = model.prior(y[idx].to(model.device), e[idx].to(model.device))
    z_sample = prior_dist.sample()
    return torch.chunk(z_sample, 2, dim=1)


def main(args):
    rng = np.random.RandomState(args.seed)
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    pl.seed_everything(args.seed)
    dataloader = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, EvalStage.TRAIN.value,
        f'version_{args.seed}', 'infer.pt')), 1, False)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    x, y, e, z = dataloader.dataset[:]
    for example_idx in range(N_EXAMPLES):
        x_seed, y_seed, e_seed, z_seed = x[[example_idx]], y[[example_idx]], e[[example_idx]], z[[example_idx]].to(
            model.device)
        zc_seed, zs_seed = torch.chunk(z_seed, 2, dim=1)
        fig, axes = plt.subplots(2, N_COLS, figsize=(2 * N_COLS, 2 * 2))
        for ax in axes.flatten():
            ax.set_xticks([])
            ax.set_yticks([])
        plot = PLOT[args.dataset]
        image_size = IMAGE_SHAPE[args.dataset]
        plot(axes[0, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 0], x_seed.reshape(image_size).detach().cpu().numpy())
        x_pred = torch.sigmoid(model.decoder.mlp(z_seed))
        plot(axes[0, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        plot(axes[1, 1], x_pred.reshape(image_size).detach().cpu().numpy())
        for col_idx in range(2, N_COLS):
            zc_sample, zs_sample = sample_prior(rng, model, y, e)
            x_pred_causal = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_sample, zs_seed))))
            x_pred_spurious = torch.sigmoid(model.decoder.mlp(torch.hstack((zc_seed, zs_sample))))
            plot(axes[0, col_idx], x_pred_causal.reshape(image_size).detach().cpu().numpy())
            plot(axes[1, col_idx], x_pred_spurious.reshape(image_size).detach().cpu().numpy())
        fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'reconstruct_from_infer')
        os.makedirs(fig_dpath, exist_ok=True)
        plt.savefig(os.path.join(fig_dpath, f'{example_idx}.png'))
        plt.close()