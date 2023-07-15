import numpy as np
import os
import pandas as pd
import torch
from utils.nn_utils import make_dataloader


def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    fpath = os.path.join(os.environ['DATA_DPATH'], 'dsprites.pt')
    if os.path.exists(fpath):
        x, y, e = torch.load(fpath)
    else:
        data = np.load(os.path.join(os.environ['DATA_DPATH'], 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
        x = data['imgs'].astype('float32')
        x = x.reshape(len(x), -1)
        factors = pd.DataFrame(data['latents_values'], columns=['color', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y'])
        # Square and ellipse, no rotation
        idxs = np.where((factors.orientation == 0) & (factors['shape'] <= 2))[0]
        x, factors = x[idxs], factors.iloc[idxs]
        n_total, x_size = x.shape
        idxs_env1 = np.where(factors['shape'] == 2)[0]

        # y is area with noise
        area = x.sum(axis=1) / x_size
        y = min_max_scale(area)
        y = np.clip(y + rng.normal(0, 0.2, size=len(y)), 0, 1)

        # Positively correlated in env0, and negatively correlated in env1
        brightness = 2 * np.copy(y) - 1
        brightness[idxs_env1] *= -1
        brightness = (brightness + 1) / 2
        brightness = np.clip(brightness + rng.normal(0, 0.2, size=len(brightness)), 0, 1)
        brightness = brightness[:, None]

        x *= brightness

        e = np.zeros(n_total)
        e[idxs_env1] = 1

        x, y, e = torch.tensor(x), torch.tensor(y), torch.tensor(e)
        y, e = y[:, None].float(), e[:, None].float()
        torch.save((x, y, e), fpath)
    n_total = len(x)
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val