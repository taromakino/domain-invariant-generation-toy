import itertools
import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from causallearn.utils.cit import CIT
from data import make_data
from model import Model
from utils.file import load_file, write


def main(args):
    existing_args = load_file(os.path.join(args.dpath, 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    # data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, existing_args.n_workers)
    # model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'))
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'), map_location='cpu')
    y, e, z = [], [], []
    with torch.no_grad():
        for x_batch, y_batch, e_batch in data_train:
            x_batch, y_batch, e_batch = x_batch.to(model.device), y_batch.to(model.device), e_batch.to(model.device)
            u_batch = torch.cat((y_batch, e_batch), dim=1)
            x_embed_batch = model.x_encoder(x_batch)
            embed_z_ux_batch = model.encoder_embed(torch.cat((x_embed_batch, u_batch), dim=1))
            z_batch = model.encoder_mu(embed_z_ux_batch)
            y.append(y_batch.cpu().numpy())
            e.append(e_batch.cpu().numpy())
            z.append(z_batch.cpu().numpy())
    y = np.concatenate(y)
    e = np.concatenate(e)
    z = np.concatenate(z)
    data = np.c_[z, y, e]
    z_idxs = [i for i in range(z.shape[1])]
    e_idxs = [i + (z_idxs[-1] + 1) for i in range(e.shape[1])]
    y_idxs = [i + (e_idxs[-1] + 1) for i in range(y.shape[1])]
    kci = CIT(data, 'kci')
    z_idx_pairs = list(itertools.combinations(z_idxs, 2))
    cause_idxs = []
    for z_i, z_j in z_idx_pairs:
        p_val_ind = kci(z_i, z_j, e_idxs)
        p_val_cind = kci(z_i, z_j, e_idxs + y_idxs)
        if p_val_cind < p_val_ind:
            cause_idxs.append(z_i)
            cause_idxs.append(z_j)
    cause_idxs = sorted(list(set(cause_idxs)))
    effect_idxs = np.setdiff1d(z_idxs, cause_idxs)
    write(os.path.join(args.dpath, 'causal_discovery.txt'), f'cause_idxs={cause_idxs}')
    write(os.path.join(args.dpath, 'causal_discovery.txt'), f'effect_idxs={effect_idxs}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--pairwise_test_name', type=str, default='RECI')
    main(parser.parse_args())