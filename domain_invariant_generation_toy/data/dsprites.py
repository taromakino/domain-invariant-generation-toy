import numpy as np
import os
import pandas as pd
import torch
from utils.nn_utils import make_dataloader
from utils.stats import min_max_scale


N_TOTAL = 10000
WIDTH_LB = 8
WIDTH_UB = 32
IMAGE_SIZE = 128


def make_raw_data():
    rng = np.random.RandomState()
    idxs_env0 = rng.choice(np.arange(N_TOTAL), N_TOTAL // 2, replace=False)
    idxs_env1 = np.setdiff1d(np.arange(N_TOTAL), idxs_env0)

    e = np.zeros(N_TOTAL)
    e[idxs_env1] = 1

    width = np.full(N_TOTAL, np.nan)
    width[idxs_env0] = (rng.normal(16, 3, len(idxs_env0))).astype(int)
    width[idxs_env1] = (rng.normal(24, 3, len(idxs_env1))).astype(int)
    width = width + width % 2
    width = np.clip(width, WIDTH_LB, WIDTH_UB)

    y = width ** 2
    y = min_max_scale(y)

    y = rng.binomial(1, y, len(y))

    brightness = np.full(N_TOTAL, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    brightness[idxs_y0_e0] = rng.normal(0.3, 0.01, len(idxs_y0_e0))
    brightness[idxs_y0_e1] = rng.normal(0.7, 0.01, len(idxs_y0_e1))
    brightness[idxs_y1_e0] = rng.normal(0.5, 0.01, len(idxs_y1_e0))
    brightness[idxs_y1_e1] = rng.normal(0.9, 0.01, len(idxs_y1_e1))
    brightness = np.clip(brightness, 0, 1)[:, None]

    center_x = rng.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TOTAL)
    center_y = rng.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TOTAL)

    x = np.zeros((N_TOTAL, IMAGE_SIZE, IMAGE_SIZE))
    for idx in range(N_TOTAL):
        half_width = width[idx] // 2
        x_lb = int(center_x[idx] - half_width)
        x_ub = int(center_x[idx] + half_width)
        y_lb = int(center_y[idx] - half_width)
        y_ub = int(center_y[idx] + half_width)
        x[idx, x_lb:x_ub, y_lb:y_ub] = brightness[idx]
    x = x.reshape(len(x), -1)
    x, y, e = torch.tensor(x), torch.tensor(y), torch.tensor(e)
    x, y, e = x.float(), y[:, None].float(), e[:, None].float()
    return e, width, y, brightness, x


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState()
    e, scale, y, brightness, x = make_raw_data()
    n_total = len(x)
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(np.arange(n_total), n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val