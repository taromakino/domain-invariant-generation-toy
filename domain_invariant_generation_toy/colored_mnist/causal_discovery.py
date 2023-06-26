import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from causallearn.search.ConstraintBased.PC import pc
from cdt.causality.pairwise import ANM, IGCI, RECI
from data import make_data
from model import Model
from utils.file import load_file, save_file


PAIRWISE_TESTS = {
    'ANM': ANM,
    'IGCI': IGCI,
    'RECI': RECI
}


def get_neighbor_names(cg, var_name, var_names, node_names):
    var_to_node_name = dict((var_name, node_names[i]) for i, var_name in enumerate(var_names))
    node_to_var_name = dict((node_name, var_names[i]) for i, node_name in enumerate(node_names))
    neighbors = cg.G.get_adjacent_nodes(cg.G.get_node(var_to_node_name[var_name]))
    return [node_to_var_name[neighbor.name] for neighbor in neighbors]


def main(args):
    pairwise_test = PAIRWISE_TESTS[args.pairwise_test_name]()
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
            u_batch = torch.cat((e_batch, y_batch), dim=1)
            z_batch = model.q_z_ux_mu(u_batch, x_batch)
            y.append(y_batch.cpu().numpy())
            z.append(z_batch.cpu().numpy())
    y = np.concatenate(y)
    z = np.concatenate(z)

    data = np.c_[y, z]
    cg = pc(data)
    node_names = cg.G.get_node_names()

    var_names = \
        ['y'] + \
        [f'z_{i}' for i in range(existing_args.z_size)]
    neighbor_names = get_neighbor_names(cg, 'y', var_names, node_names)

    parent_names = []
    for neighbor_name in neighbor_names:
        neighbor_col_idx = var_names.index(neighbor_name)
        if pairwise_test.predict_proba((data[:, neighbor_col_idx], y.astype('int'))) > 0:
            parent_names.append(neighbor_name)
    parent_names = set(parent_names)
    save_file(parent_names, os.path.join(args.dpath, 'parent_names.pkl'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--pairwise_test_name', type=str, default='ANM')
    main(parser.parse_args())