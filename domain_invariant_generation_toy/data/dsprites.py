import matplotlib.pyplot as plt
import numpy as np
import torch
from utils.nn_utils import make_dataloader
from utils.plot import hist_discrete
from utils.stats import min_max_scale


RNG = np.random.RandomState(0)
N_TRAINVAL = 10000
N_TEST = 2000
WIDTH_LB = 8
WIDTH_UB = 32
IMAGE_SIZE = 64


def make_trainval_data():
    idxs_env0 = RNG.choice(np.arange(N_TRAINVAL), N_TRAINVAL // 2, replace=False)
    idxs_env1 = np.setdiff1d(np.arange(N_TRAINVAL), idxs_env0)

    e = np.zeros(N_TRAINVAL)
    e[idxs_env1] = 1

    width = np.full(N_TRAINVAL, np.nan)
    width[idxs_env0] = (RNG.normal(18, 4, len(idxs_env0))).astype(int)
    width[idxs_env1] = (RNG.normal(22, 4, len(idxs_env1))).astype(int)
    width = np.clip(width, WIDTH_LB, WIDTH_UB)

    y = min_max_scale(width ** 2)
    y = RNG.binomial(1, y, len(y))

    brightness = np.full(N_TRAINVAL, np.nan)
    idxs_y0_e0 = np.where((y == 0) & (e == 0))[0]
    idxs_y1_e0 = np.where((y == 1) & (e == 0))[0]
    idxs_y0_e1 = np.where((y == 0) & (e == 1))[0]
    idxs_y1_e1 = np.where((y == 1) & (e == 1))[0]
    brightness[idxs_y0_e0] = RNG.normal(0.2, 0.1, len(idxs_y0_e0))
    brightness[idxs_y1_e0] = RNG.normal(0.4, 0.1, len(idxs_y1_e0))
    brightness[idxs_y0_e1] = RNG.normal(0.6, 0.1, len(idxs_y0_e1))
    brightness[idxs_y1_e1] = RNG.normal(0.8, 0.1, len(idxs_y1_e1))
    brightness = np.clip(brightness, 0, 1)[:, None]

    center_x = RNG.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TRAINVAL)
    center_y = RNG.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TRAINVAL)

    x = np.zeros((N_TRAINVAL, IMAGE_SIZE, IMAGE_SIZE))
    for idx in range(N_TRAINVAL):
        half_width_floor = np.floor(width[idx] / 2)
        half_width_ceil = np.ceil(width[idx] / 2)
        x_lb = int(center_x[idx] - half_width_floor)
        x_ub = int(center_x[idx] + half_width_ceil)
        y_lb = int(center_y[idx] - half_width_floor)
        y_ub = int(center_y[idx] + half_width_ceil)
        x[idx, x_lb:x_ub, y_lb:y_ub] = brightness[idx]
    x = x.reshape(len(x), -1)
    x, y, e = torch.tensor(x), torch.tensor(y), torch.tensor(e)
    x, y, e = x.float(), y[:, None].float(), e[:, None].float()
    return e, width, y, brightness, x


def make_test_data(batch_size):
    idxs_lhs = RNG.choice(np.arange(N_TEST), N_TEST // 2, replace=False)
    idxs_rhs = np.setdiff1d(np.arange(N_TEST), idxs_lhs)

    width = np.full(N_TEST, np.nan)
    width[idxs_lhs] = (RNG.normal(18, 4, len(idxs_lhs))).astype(int)
    width[idxs_rhs] = (RNG.normal(22, 4, len(idxs_rhs))).astype(int)
    width = np.clip(width, WIDTH_LB, WIDTH_UB)

    y = min_max_scale(width ** 2)
    y = RNG.binomial(1, y, len(y))

    brightness = np.full(N_TEST, np.nan)
    idxs_y0 = np.where(y == 0)[0]
    idxs_y1 = np.where(y == 1)[0]
    brightness[idxs_y0] = RNG.normal(0.8, 0.01, len(idxs_y0))
    brightness[idxs_y1] = RNG.normal(0.2, 0.01, len(idxs_y1))
    brightness = np.clip(brightness, 0, 1)[:, None]

    center_x = RNG.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TRAINVAL)
    center_y = RNG.randint(WIDTH_UB // 2, IMAGE_SIZE - WIDTH_UB // 2 + 1, N_TRAINVAL)

    x = np.zeros((N_TEST, IMAGE_SIZE, IMAGE_SIZE))
    for idx in range(N_TEST):
        half_width_floor = np.floor(width[idx] / 2)
        half_width_ceil = np.ceil(width[idx] / 2)
        x_lb = int(center_x[idx] - half_width_floor)
        x_ub = int(center_x[idx] + half_width_ceil)
        y_lb = int(center_y[idx] - half_width_floor)
        y_ub = int(center_y[idx] + half_width_ceil)
        x[idx, x_lb:x_ub, y_lb:y_ub] = brightness[idx]
    x = x.reshape(len(x), -1)
    x, y = torch.tensor(x), torch.tensor(y)
    x, y = x.float(), y[:, None].float()
    return make_dataloader((x, y), batch_size, False)


def make_data(train_ratio, batch_size):
    e, width, y, brightness, x = make_trainval_data()
    n_total = len(x)
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
    e, width, y, brightness, x = make_trainval_data()
    e, y, brightness = e.squeeze(), y.squeeze(), brightness.squeeze()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    hist_discrete(axes[0], width[e == 0])
    hist_discrete(axes[1], width[e == 1])
    axes[0].set_title('p(width|e=0)')
    axes[1].set_title('p(width|e=1)')
    fig.suptitle('Assumed Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    axes[0].hist(brightness[(y == 0) & (e == 0)], bins='auto')
    axes[1].hist(brightness[(y == 1) & (e == 0)], bins='auto')
    axes[2].hist(brightness[(y == 0) & (e == 1)], bins='auto')
    axes[3].hist(brightness[(y == 1) & (e == 1)], bins='auto')
    fig.suptitle('Assumed Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    hist_discrete(axes[0], width[(y == 0) & (e == 0)])
    hist_discrete(axes[1], width[(y == 1) & (e == 0)])
    hist_discrete(axes[2], width[(y == 0) & (e == 1)])
    hist_discrete(axes[3], width[(y == 1) & (e == 1)])
    axes[0].set_title('p(width|y=0,e=0)')
    axes[1].set_title('p(width|y=1,e=0)')
    axes[2].set_title('p(width|y=0,e=1)')
    axes[3].set_title('p(width|y=1,e=1)')
    fig.suptitle('Assumed Non-Gaussian')
    fig.tight_layout()
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].hist(brightness[e == 0], bins='auto')
    axes[1].hist(brightness[e == 1], bins='auto')
    axes[0].set_title('p(brightness|e=0)')
    axes[1].set_title('p(brightness|e=1)')
    fig.suptitle('Assumed Non-Gaussian')
    fig.tight_layout()
    plt.show(block=True)


if __name__ == '__main__':
    main()