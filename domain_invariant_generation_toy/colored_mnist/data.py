import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


PROB_ZERO_E0 = 0.1


def flip_binary(rng, x, flip_prob):
    idxs = np.arange(len(x))
    flip_idxs = rng.choice(idxs, size=int(flip_prob * len(idxs)), replace=False)
    x[flip_idxs] = 1 - x[flip_idxs]
    return x


def make_raw_data():
    rng = np.random.RandomState(0)
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    images, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_total = len(images)
    idxs_env0 = []
    zero_idxs = np.where(digits == 0)[0]
    one_idxs = np.where(digits == 1)[0]
    idxs_env0.append(rng.choice(zero_idxs, size=int(PROB_ZERO_E0 * len(zero_idxs))))
    idxs_env0.append(rng.choice(one_idxs, size=int((1 - PROB_ZERO_E0) * len(one_idxs))))
    idxs_env0 = np.concatenate(idxs_env0)
    idxs_env1 = np.setdiff1d(np.arange(n_total), idxs_env0)

    e = torch.zeros(n_total)
    e[idxs_env1] = 1

    y = flip_binary(rng, digits.clone(), 0.25)

    colors = np.full(n_total, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    colors[idxs_y0_e0] = rng.normal(0.2, 0.01, len(idxs_y0_e0))
    colors[idxs_y1_e0] = rng.normal(0.6, 0.01, len(idxs_y1_e0))
    colors[idxs_y0_e1] = rng.normal(0.4, 0.01, len(idxs_y0_e1))
    colors[idxs_y1_e1] = rng.normal(0.8, 0.01, len(idxs_y1_e1))
    colors = np.clip(colors, 0, 1)[:, None, None]

    images = torch.stack([images, images], dim=1)
    images = images / 255
    images[:, 0, :, :] *= colors
    images[:, 1, :, :] *= (1 - colors)
    x = images.flatten(start_dim=1)

    y, e = y[:, None].float(), e[:, None]
    return e, digits, y, colors, x


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    e, digits, y, colors, x = make_raw_data()
    n_total = len(e)
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val