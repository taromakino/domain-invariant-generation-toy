import numpy as np
import os
import pandas as pd
import torch
from utils.nn_utils import make_dataloader
from utils.stats import min_max_scale


PROB_SCALE_ENV0 = {
    0.5: 0.25,
    0.6: 1,
    0.7: 0.25
}


PROB_SCALE_ENV1 = {
    0.8: 0.25,
    0.9: 1,
    1.0: 0.25
}


def make_raw_data():
    rng = np.random.RandomState(0)
    data = np.load(os.path.join(os.environ['DATA_DPATH'], 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz'))
    images = data['imgs'].astype('float32')
    images = images.reshape(len(images), -1)
    factors = pd.DataFrame(data['latents_values'], columns=['color', 'shape', 'scale', 'orientation', 'pos_x', 'pos_y'])
    # No rotation, square only
    idxs = np.where((factors.orientation == 0) & (factors['shape'] == 1))[0]
    images, factors = images[idxs], factors.iloc[idxs]

    # Assumes the dictionaries' keys do not overlap
    idxs_env0, idxs_env1 = [], []
    for value, prob in PROB_SCALE_ENV0.items():
        subset_idxs = np.where(factors.scale == value)[0]
        subset_idxs = rng.choice(subset_idxs, int(prob * len(subset_idxs)), replace=False)
        idxs_env0.append(subset_idxs)
    for value, prob in PROB_SCALE_ENV1.items():
        subset_idxs = np.where(factors.scale == value)[0]
        subset_idxs = rng.choice(subset_idxs, int(prob * len(subset_idxs)), replace=False)
        idxs_env1.append(subset_idxs)
    idxs_env0, idxs_env1 = np.concatenate(idxs_env0), np.concatenate(idxs_env1)
    idxs = np.concatenate((idxs_env0, idxs_env1))
    images, factors = images[idxs], factors.iloc[idxs]
    n_total, x_size = images.shape

    e = np.concatenate((np.zeros(len(idxs_env0)), np.ones(len(idxs_env1))))

    y = images.sum(axis=1) / x_size
    y = min_max_scale(y)
    y = rng.binomial(1, y, len(y))

    brightness = np.full(n_total, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    brightness[idxs_y0_e0] = rng.normal(0.3, 0.01, len(idxs_y0_e0))
    brightness[idxs_y0_e1] = rng.normal(0.7, 0.01, len(idxs_y0_e1))
    brightness[idxs_y1_e0] = rng.normal(0.5, 0.01, len(idxs_y1_e0))
    brightness[idxs_y1_e1] = rng.normal(0.9, 0.01, len(idxs_y1_e1))
    brightness = np.clip(brightness, 0, 1)[:, None]

    x = images * brightness

    x, y, e = torch.tensor(x), torch.tensor(y), torch.tensor(e)
    x, y, e = x.float(), y[:, None].float(), e[:, None].float()
    return e, factors.scale.values, y, brightness, x


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    e, scale, y, brightness, x = make_raw_data()
    n_total = len(x)
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val