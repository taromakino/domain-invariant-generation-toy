import numpy as np
import itertools
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from causallearn.utils.KCI import KCI
from data import make_data
from model import Model
from utils.file import load_file, write


def get_neighbor_names(cg, var_name, var_names, node_names):
    var_to_node_name = dict((var_name, node_names[i]) for i, var_name in enumerate(var_names))
    node_to_var_name = dict((node_name, var_names[i]) for i, node_name in enumerate(node_names))
    neighbors = cg.G.get_adjacent_nodes(cg.G.get_node(var_to_node_name[var_name]))
    return [node_to_var_name[neighbor.name] for neighbor in neighbors]


def main(args):
    existing_args = load_file(os.path.join(args.dpath, 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    # data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, existing_args.n_workers)
    # model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'))
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, 1)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'), map_location='cpu')
    y, z = [], []
    with torch.no_grad():
        for x_batch, y_batch, e_batch in data_train:
            x_batch, y_batch, e_batch = x_batch.to(model.device), y_batch.to(model.device), e_batch.to(model.device)
            u_batch = torch.cat((y_batch, e_batch), dim=1)
            z_batch = model.encoder_mu(u_batch, x_batch)
            y.append(y_batch.cpu().numpy())
            z.append(z_batch.cpu().numpy())
    y = np.concatenate(y)
    z = np.concatenate(z)

    # pairs = list(itertools.combinations(np.arange(existing_args.z_size), 2))
    # for i, j in pairs:

    # write(os.path.join(args.dpath, 'causal_discovery.txt'), f'cause_idxs={cause_idxs}')
    # write(os.path.join(args.dpath, 'causal_discovery.txt'), f'effect_idxs={effect_idxs}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--pairwise_test_name', type=str, default='RECI')
    main(parser.parse_args())