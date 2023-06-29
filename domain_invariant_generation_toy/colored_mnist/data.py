import numpy as np
import os
import torch
from scipy.special import expit as sigmoid
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets


def make_dataloader(data_tuple, batch_size, n_workers, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size, num_workers=n_workers,
        pin_memory=True, persistent_workers=True)


def make_environment(rng, images, digits, temperature, e_arr):
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    n = len(images)
    y = np.full(n, np.nan)
    binary_digits = (digits < 5).int()
    for binary_digit in [0, 1]:
        idxs = np.where(binary_digits == binary_digit)[0]
        y_noise = rng.beta(2, 2, len(idxs)) - 0.5
        y[idxs] = 0.25 * (2 * binary_digit - 1) + y_noise
    y = torch.tensor(y, dtype=torch.float32)
    z_s = torch.bernoulli(sigmoid(y / temperature)) # Color
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - z_s).int(), :, :] *= 0
    images = images / 255
    images = images.flatten(start_dim=1)
    e_arr = torch.repeat_interleave(torch.tensor(e_arr, dtype=torch.float32)[None], len(images), dim=0)
    return {
        'x': images,
        'y': y[:, None],
        'e': e_arr
    }


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    mnist_train = (mnist.data[:50000], mnist.targets[:50000])
    mnist_val = (mnist.data[50000:], mnist.targets[50000:])
    envs = [
        make_environment(rng, mnist_train[0][::2], mnist_train[1][::2], 0.1, [1, 0, 0]),
        make_environment(rng, mnist_train[0][1::2], mnist_train[1][1::2], 0.01, [0, 1, 0]),
        make_environment(rng, mnist_val[0], mnist_val[1], -0.01, [0, 0, 1])
    ]
    x_trainval = torch.cat((envs[0]['x'], envs[1]['x']))
    y_trainval = torch.cat((envs[0]['y'], envs[1]['y']))
    e_trainval = torch.cat((envs[0]['e'], envs[1]['e']))
    n_trainval = len(x_trainval)
    n_train = int(train_ratio * n_trainval)
    train_idxs = rng.choice(n_trainval, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_trainval), train_idxs)
    x_train, y_train, e_train = x_trainval[train_idxs], y_trainval[train_idxs], e_trainval[train_idxs]
    x_val, y_val, e_val = x_trainval[val_idxs], y_trainval[val_idxs], e_trainval[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val