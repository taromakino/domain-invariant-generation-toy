import numpy as np
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from cdt.causality.pairwise import ANM, IGCI, RECI
from data import make_data
from model import Model
from utils.file import load_file, write


PAIRWISE_TESTS = {
    'ANM': ANM,
    'IGCI': IGCI,
    'RECI': RECI
}


def main(args):
    pairwise_test = PAIRWISE_TESTS[args.pairwise_test_name]()
    existing_args = load_file(os.path.join(args.dpath, 'args.pkl'))
    pl.seed_everything(existing_args.seed)
    data_train, data_val = make_data(existing_args.train_ratio, existing_args.batch_size, existing_args.n_workers)
    model = Model.load_from_checkpoint(os.path.join(args.dpath, 'checkpoints', 'best.ckpt'))
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

    z_idxs = [i for i in range(existing_args.z_size)]
    cause_idxs = []
    for z_idx in z_idxs:
        if pairwise_test.predict_proba((z[:, z_idx], y)) > 0:
            cause_idxs.append(z_idx)
    effect_idxs = np.setdiff1d(z_idxs, cause_idxs)
    write(os.path.join(args.dpath, 'causal_discovery.txt'), f'cause_idxs={cause_idxs}')
    write(os.path.join(args.dpath, 'causal_discovery.txt'), f'effect_idxs={effect_idxs}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--pairwise_test_name', type=str, default='RECI')
    main(parser.parse_args())