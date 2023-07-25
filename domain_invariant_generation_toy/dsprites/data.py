import numpy as np
import os
import pandas as pd
import torch
from utils.nn_utils import make_dataloader
from utils.stats import min_max_scale


PROB_SHAPE_E0 = [0.1, 0.6, 0.8]


def make_raw_data():
    rng = np.random.RandomState(0)
    data = np.load(os.path.join(os.environ['DATA_DPATH'], 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
    images = data['imgs'].astype('float32')
    images = images.reshape(len(images), -1)
    factors = pd.DataFrame(data['latents_values'], columns=['color', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y'])
    orientation = factors.orientation.values
    idxs = np.where(orientation == 0)[0]
    images, factors = images[idxs], factors.iloc[idxs]

    shapes = factors['shape'].values
    n_shapes = int(shapes.max())
    idxs_env0, idxs_env1 = [], []
    for shape_idx in range(n_shapes):
        idxs = np.where(shapes == shape_idx + 1)[0]
        idxs_e0_elem = rng.choice(idxs, size=int(PROB_SHAPE_E0[shape_idx] * len(idxs)), replace=False)
        idxs_e1_elem = np.setdiff1d(idxs, idxs_e0_elem)
        idxs_env0.append(idxs_e0_elem)
        idxs_env1.append(idxs_e1_elem)
    idxs_env0 = np.concatenate(idxs_env0)
    idxs_env1 = np.concatenate(idxs_env1)

    n_total, x_size = images.shape

    y = images.sum(axis=1) / x_size
    y += rng.normal(0, 0.01, size=len(y))
    y = min_max_scale(y)

    y_q1 = np.quantile(y, 0.25)
    brightness_env0 = np.full(len(idxs_env0), np.nan)
    brightness_env1 = np.full(len(idxs_env1), np.nan)
    y_env0 = y[idxs_env0]
    y_env1 = y[idxs_env1]

    brightness_env0[y_env0 < y_q1] = rng.normal(0.25, 0.01, (y_env0 < y_q1).sum())
    brightness_env0[y_env0 >= y_q1] = rng.normal(0.75, 0.01, (y_env0 >= y_q1).sum())
    brightness_env1[y_env1 < y_q1] = rng.normal(0.75, 0.01, (y_env1 < y_q1).sum())
    brightness_env1[y_env1 >= y_q1] = rng.normal(0.25, 0.01, (y_env1 >= y_q1).sum())
    brightness = np.full_like(y, np.nan)
    brightness[idxs_env0] = brightness_env0
    brightness[idxs_env1] = brightness_env1

    brightness = np.clip(brightness, 0, 1)[:, None]

    images *= brightness

    e = np.zeros(n_total)
    e[idxs_env1] = 1
    x, y, e = torch.tensor(images), torch.tensor(y), torch.tensor(e)
    y, e = y[:, None].float(), e[:, None].float()
    return e, shapes, y, brightness, x


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    e, shapes, y, brightness, x = make_raw_data()
    n_total = len(x)
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val