import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


def flip_binary(rng, x, flip_prob):
    idxs = np.arange(len(x))
    flip_idxs = rng.choice(idxs, size=int(flip_prob * len(idxs)), replace=False)
    x[flip_idxs] = 1 - x[flip_idxs]
    return x


def make_environment(rng, images, digits, color_flip_prob):
    y = flip_binary(rng, digits.clone(), 0.25)
    colors = flip_binary(rng, y.clone(), color_flip_prob)
    images = torch.stack([images, images], dim=1)
    images[np.arange(len(images)), 1 - colors, :, :] = 0
    images = images / 255
    images = images.flatten(start_dim=1)
    return digits, y, colors, images


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    images, digits = mnist.data[binary_idxs], mnist.targets[binary_idxs]
    n_total = len(images)
    idxs = np.arange(n_total)
    rng.shuffle(idxs)
    idxs_0 = idxs[:int(n_total / 2)]
    idxs_1 = idxs[int(n_total / 2):]
    digits_0, y_0, colors_0, images_0 = make_environment(rng, images[idxs_0], digits[idxs_0], 0.1)
    digits_1, y_1, colors_1, images_1 = make_environment(rng, images[idxs_1], digits[idxs_1], 0.9)
    e = torch.zeros(n_total)[:, None]
    e[idxs_1] = 1
    digits = torch.cat((digits_0, digits_1))[:, None].float()
    y = torch.cat((y_0, y_1))[:, None].float()
    colors = torch.cat((colors_0, colors_1))[:, None].float()
    images = torch.cat((images_0, images_1))
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    e_train, digits_train, y_train, colors_train, images_train = e[train_idxs], digits[train_idxs], y[train_idxs], \
        colors[train_idxs], images[train_idxs]
    e_val, digits_val, y_val, colors_val, images_val = e[val_idxs], digits[val_idxs], y[val_idxs], colors[val_idxs], \
        images[val_idxs]
    data_train = make_dataloader((e_train, digits_train, y_train, colors_train, images_train), batch_size, n_workers, True)
    data_val = make_dataloader((e_val, digits_val, y_val, colors_val, images_val), batch_size, n_workers, False)
    return data_train, data_val