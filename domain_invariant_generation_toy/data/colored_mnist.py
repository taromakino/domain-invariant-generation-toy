import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


RNG = np.random.RandomState(0)
PROB_ZERO_E0 = 0.25


def flip_binary(x, flip_prob):
    idxs = np.arange(len(x))
    flip_idxs = RNG.choice(idxs, size=int(flip_prob * len(idxs)), replace=False)
    x[flip_idxs] = 1 - x[flip_idxs]
    return x


def make_trainval_data():
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    images, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_total = len(images)

    idxs_env0 = []
    zero_idxs = np.where(digits == 0)[0]
    one_idxs = np.where(digits == 1)[0]
    idxs_env0.append(RNG.choice(zero_idxs, size=int(PROB_ZERO_E0 * len(zero_idxs))))
    idxs_env0.append(RNG.choice(one_idxs, size=int((1 - PROB_ZERO_E0) * len(one_idxs))))
    idxs_env0 = np.concatenate(idxs_env0)
    idxs_env1 = np.setdiff1d(np.arange(n_total), idxs_env0)

    e = torch.zeros(n_total)
    e[idxs_env1] = 1

    y = flip_binary(digits.clone(), 0.25)

    colors = np.full(n_total, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    colors[idxs_y0_e0] = RNG.normal(0.2, 0.1, len(idxs_y0_e0))
    colors[idxs_y1_e0] = RNG.normal(0.4, 0.1, len(idxs_y1_e0))
    colors[idxs_y0_e1] = RNG.normal(0.6, 0.1, len(idxs_y0_e1))
    colors[idxs_y1_e1] = RNG.normal(0.8, 0.1, len(idxs_y1_e1))
    colors = np.clip(colors, 0, 1)[:, None, None]

    images = torch.stack([images, images], dim=1)
    images = images / 255
    images[:, 0, :, :] *= colors
    images[:, 1, :, :] *= (1 - colors)
    x = images.flatten(start_dim=1)

    y, e = y[:, None].float(), e[:, None]
    return e, digits, y, colors, x


def make_test_data(batch_size):
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=False, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    images, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_total = len(images)

    y = flip_binary(digits.clone(), 0.25)

    colors = np.full(n_total, np.nan)
    idxs_y0 = np.where(y == 0)[0]
    idxs_y1 = np.where(y == 1)[0]
    colors[idxs_y0] = RNG.normal(0.8, 0.1, len(idxs_y0))
    colors[idxs_y1] = RNG.normal(0.2, 0.1, len(idxs_y1))
    colors = np.clip(colors, 0, 1)[:, None, None]

    images = torch.stack([images, images], dim=1)
    images = images / 255
    images[:, 0, :, :] *= colors
    images[:, 1, :, :] *= (1 - colors)
    x = images.flatten(start_dim=1)
    y = y[:, None].float()
    return make_dataloader((x, y), batch_size, False)


def make_data(train_ratio, batch_size):
    e, digits, y, colors, x = make_trainval_data()
    n_total = len(e)
    n_train = int(train_ratio * n_total)
    train_idxs = RNG.choice(np.arange(n_total), n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, False)
    data_test = make_test_data(batch_size)
    return data_train, data_val, data_test


def main():
    e, digits, y, colors, x = make_trainval_data()
    e, y, colors = e.squeeze(), y.squeeze(), colors.squeeze()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(digits[e == 0])
    axes[1].hist(digits[e == 1])
    axes[0].set_title('p(digit|e=0)')
    axes[1].set_title('p(digit|e=1)')
    fig.suptitle('Should be Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(colors[(y == 0) & (e == 0)])
    axes[1].hist(colors[(y == 1) & (e == 0)])
    axes[2].hist(colors[(y == 0) & (e == 1)])
    axes[3].hist(colors[(y == 1) & (e == 1)])
    axes[0].set_title('p(color|y=0,e=0)')
    axes[1].set_title('p(color|y=1,e=0)')
    axes[2].set_title('p(color|y=0,e=1)')
    axes[3].set_title('p(color|y=1,e=1)')
    fig.suptitle('Should be Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(digits[(y == 0) & (e == 0)])
    axes[1].hist(digits[(y == 1) & (e == 0)])
    axes[2].hist(digits[(y == 0) & (e == 1)])
    axes[3].hist(digits[(y == 1) & (e == 1)])
    axes[0].set_title('p(digit|y=0,e=0)')
    axes[1].set_title('p(digit|y=1,e=0)')
    axes[2].set_title('p(digit|y=0,e=1)')
    axes[3].set_title('p(digit|y=1,e=1)')
    fig.suptitle('Should be non-Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(colors[e == 0])
    axes[1].hist(colors[e == 1])
    axes[0].set_title('p(color|e=0)')
    axes[1].set_title('p(color|e=1)')
    fig.suptitle('Should be non-Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(colors[y == 0])
    axes[1].hist(colors[y == 1])
    axes[0].set_title('p(color|y=0)')
    axes[1].set_title('p(color|y=1)')
    fig.suptitle('Should not be perfectly predictive')
    fig.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    main()