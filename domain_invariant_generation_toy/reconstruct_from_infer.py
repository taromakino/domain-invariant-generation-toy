import matplotlib.pyplot as plt
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import PLOT, IMAGE_SHAPE
from models.vae import VAE
from utils.enums import Task, EvalStage
from utils.file import load_file
from utils.nn_utils import make_dataloader


def main(args):
    task_dpath = os.path.join(args.dpath, Task.VAE.value)
    existing_args = load_file(os.path.join(task_dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, args.eval_stage.value,
        f'version_{args.seed}', 'infer.pt')), 1, False)
    model = VAE.load_from_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    x_full, y_full, z_full = data_train.dataset[:]
    fig, axes = plt.subplots(2, args.n_examples, figsize=(2 * args.n_examples, 2 * 2))
    for ax in axes.flatten():
        ax.set_xticks([])
        ax.set_yticks([])
    for example_idx in range(args.n_examples):
        x = x_full[[example_idx]].to(model.device)
        z = z_full[[example_idx]].to(model.device)
        plot = PLOT[existing_args.dataset]
        image_size = IMAGE_SHAPE[existing_args.dataset]
        plot(axes[0, example_idx], x.reshape(image_size).detach().cpu().numpy())
        x_pred = torch.sigmoid(model.decoder.mlp(z))
        plot(axes[1, example_idx], x_pred.reshape(image_size).detach().cpu().numpy())
    fig_dpath = os.path.join(task_dpath, f'version_{args.seed}', 'fig', 'generate_inferred_z')
    os.makedirs(fig_dpath, exist_ok=True)
    plt.savefig(os.path.join(fig_dpath, f'{args.eval_stage.value}.png'))
    plt.close()


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage), default='train')
    parser.add_argument('--n_examples', type=int, default=10)
    main(parser.parse_args())