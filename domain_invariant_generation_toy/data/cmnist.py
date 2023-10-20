import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


RNG = np.random.RandomState(0)
PROB_ZERO_E0 = 0.25
SPURIOUS_SD = 0.05
X_SIZE = 2 * 28 ** 2


def flip_binary(x, flip_prob):
    idxs = np.arange(len(x))
    flip_idxs = RNG.choice(idxs, size=int(flip_prob * len(idxs)), replace=False)
    x[flip_idxs] = 1 - x[flip_idxs]
    return x


def make_trainval_data():
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    x, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_trainval = len(x)

    y = flip_binary(digits.clone(), 0.25)

    idxs_env0 = []
    zero_idxs = np.where(digits == 0)[0]
    one_idxs = np.where(digits == 1)[0]
    idxs_env0.append(RNG.choice(zero_idxs, size=int(PROB_ZERO_E0 * len(zero_idxs))))
    idxs_env0.append(RNG.choice(one_idxs, size=int((1 - PROB_ZERO_E0) * len(one_idxs))))
    idxs_env0 = np.concatenate(idxs_env0)
    idxs_env1 = np.setdiff1d(np.arange(n_trainval), idxs_env0)

    e = torch.zeros(n_trainval, dtype=torch.long)
    e[idxs_env1] = 1

    colors = np.full(n_trainval, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    colors[idxs_y0_e0] = RNG.normal(0.2, SPURIOUS_SD, len(idxs_y0_e0))
    colors[idxs_y1_e0] = RNG.normal(0.4, SPURIOUS_SD, len(idxs_y1_e0))
    colors[idxs_y0_e1] = RNG.normal(0.8, SPURIOUS_SD, len(idxs_y0_e1))
    colors[idxs_y1_e1] = RNG.normal(0.6, SPURIOUS_SD, len(idxs_y1_e1))
    colors = np.clip(colors, 0, 1)[:, None, None]

    x = torch.stack([x, x], dim=1)
    x = x / 255
    x[:, 0, :, :] *= colors
    x[:, 1, :, :] *= (1 - colors)
    x = x.flatten(start_dim=1)

    y = y.long()
    e = e
    c = digits.float()
    s = torch.tensor(colors.squeeze()).float()
    return x, y, e, c, s


def make_ood_data(is_test):
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=False, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    x, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_total = len(x)
    test_idxs = RNG.choice(n_total, n_total // 2, replace=False)
    val_ood_idxs = np.setdiff1d(np.arange(n_total), test_idxs)
    if is_test:
        x, digits = x[test_idxs], digits[test_idxs]
    else:
        x, digits = x[val_ood_idxs], digits[val_ood_idxs]

    y = flip_binary(digits.clone(), 0.25)

    mu = 0.9 if is_test else 0.1
    colors = RNG.normal(mu, SPURIOUS_SD, n_total)[:, None, None]

    x = torch.stack([x, x], dim=1)
    x = x / 255
    x[:, 0, :, :] *= colors
    x[:, 1, :, :] *= (1 - colors)
    x = x.flatten(start_dim=1)

    y = y.long()
    e = torch.full_like(y, np.nan, dtype=torch.float32)
    c = digits.float()
    s = torch.tensor(colors.squeeze()).float()
    return x, y, e, c, s


def make_data(train_ratio, batch_size):
    x, y, e, c, s = make_trainval_data()
    n_total = len(e)
    n_train = int(train_ratio * n_total)
    train_idxs = RNG.choice(np.arange(n_total), n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    x_train, y_train, e_train, c_train, s_train = x[train_idxs], y[train_idxs], e[train_idxs], c[train_idxs], s[train_idxs]
    x_val_id, y_val_id, e_val_id, c_val_id, s_val_id = x[val_idxs], y[val_idxs], e[val_idxs], c[val_idxs], s[val_idxs]
    x_val_ood, y_val_ood, e_val_ood, c_val_ood, s_val_ood = make_ood_data(False)
    x_test, y_test, e_test, c_test, s_test = make_ood_data(True)
    data_train = make_dataloader((x_train, y_train, e_train, c_train, s_train), batch_size, True)
    data_val_id = make_dataloader((x_val_id, y_val_id, e_val_id, c_val_id, s_val_id), batch_size, False)
    data_val_ood = make_dataloader((x_val_ood, y_val_ood, e_val_ood, c_val_ood, s_val_ood), batch_size, False)
    data_test = make_dataloader((x_test, y_test, e_test, c_test, s_test), batch_size, False)
    return data_train, data_val_id, data_val_ood, data_test


def main():
    x, y, e, c, s = make_trainval_data()
    digits = c
    colors = s
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(digits[e == 0])
    axes[1].hist(digits[e == 1])
    axes[0].set_title('p(digit|e=0)')
    axes[1].set_title('p(digit|e=1)')
    fig.suptitle('Should be Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(colors[(y == 0) & (e == 0)], bins='auto')
    axes[1].hist(colors[(y == 0) & (e == 1)], bins='auto')
    axes[2].hist(colors[(y == 1) & (e == 0)], bins='auto')
    axes[3].hist(colors[(y == 1) & (e == 1)], bins='auto')
    axes[0].set_title('p(color|y=0,e=0)')
    axes[1].set_title('p(color|y=0,e=1)')
    axes[2].set_title('p(color|y=1,e=0)')
    axes[3].set_title('p(color|y=1,e=1)')
    fig.suptitle('Should be Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(digits[(y == 0) & (e == 0)])
    axes[1].hist(digits[(y == 0) & (e == 1)])
    axes[2].hist(digits[(y == 1) & (e == 0)])
    axes[3].hist(digits[(y == 1) & (e == 1)])
    axes[0].set_title('p(digit|y=0,e=0)')
    axes[1].set_title('p(digit|y=0,e=1)')
    axes[2].set_title('p(digit|y=1,e=0)')
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
    plt.show(block=True)


if __name__ == '__main__':
    main()