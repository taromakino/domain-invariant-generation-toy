import numpy as np
import os
import pandas as pd
import torch
from utils.nn_utils import make_dataloader


def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())


N_SHAPES = 3
P_SHAPE_E0 = [0.75, 0.5, 0.25]
Y_NOISE_SD = 0.1
BRIGHTNESS_NOISE_SD = [0.2, 0.3]


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    data = np.load(os.path.join(os.environ['DATA_DPATH'], 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
    x = data['imgs'].astype('float32')
    x = x.reshape(len(x), -1)
    factors = pd.DataFrame(data['latents_values'], columns=['color', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y'])
    shape = factors['shape'].values

    idxs_e0, idxs_e1 = [], []
    for shape_idx in range(N_SHAPES):
        idxs = np.where(shape == shape_idx + 1)[0]
        idxs_e0_elem = rng.choice(idxs, size=int(P_SHAPE_E0[shape_idx] * len(idxs)), replace=False)
        idxs_e1_elem = np.setdiff1d(idxs, idxs_e0_elem)
        idxs_e0.append(idxs_e0_elem)
        idxs_e1.append(idxs_e1_elem)
    idxs_e0 = np.concatenate(idxs_e0)
    idxs_e1 = np.concatenate(idxs_e1)

    n_examples, x_size = x.shape
    # Standardize and add e-invariant noise
    area = x.sum(axis=1) / x_size
    area = (area - area.mean()) / area.std()
    y = (area + rng.normal(0, Y_NOISE_SD, len(area))).astype('float32')
    # Add e-dependent noise and scale to [0, 1]
    brightness = np.copy(y)
    brightness[idxs_e0] += rng.normal(0, BRIGHTNESS_NOISE_SD[0], len(idxs_e0))
    brightness[idxs_e1] += rng.normal(0, BRIGHTNESS_NOISE_SD[1], len(idxs_e1))
    brightness[idxs_e0] = min_max_scale(brightness[idxs_e0])
    brightness[idxs_e1] = min_max_scale(brightness[idxs_e1])

    brightness = brightness[:, None]

    x[idxs_e0] *= brightness[idxs_e0]
    x[idxs_e1] *= brightness[idxs_e1]

    e = np.zeros(n_examples, dtype='float32')
    e[idxs_e1] = 1

    x, y, e = torch.tensor(x), torch.tensor(y), torch.tensor(e)
    y, e = y[:, None], e[:, None]

    n_train = int(train_ratio * n_examples)
    train_idxs = rng.choice(n_examples, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_examples), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val