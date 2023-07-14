import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


N_CLASSES = 2
N_ENVS = 2


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
    env0_idxs = rng.choice(np.arange(n_total), int(train_ratio * n_total), replace=False)
    env1_idxs = np.setdiff1d(np.arange(n_total), env0_idxs)
    digits_env0, y_env0, colors_env0, images_env0 = make_environment(rng, images[env0_idxs], digits[env0_idxs], 0.25)
    digits_env1, y_env1, colors_env1, images_env1 = make_environment(rng, images[env1_idxs], digits[env1_idxs], 0.75)
    e0 = torch.zeros(len(env0_idxs))
    e1 = torch.ones(len(env1_idxs))
    e = torch.cat((e0, e1))[:, None]
    digits = torch.cat((digits_env0, digits_env1)).float()[:, None]
    y = torch.cat((y_env0, y_env1)).float()[:, None]
    colors = torch.cat((colors_env0, colors_env1)).float()[:, None]
    images = torch.cat((images_env0, images_env1))
    n_train = int(train_ratio * n_total)
    train_idxs = rng.choice(n_total, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_total), train_idxs)
    e_train, digits_train, y_train, colors_train, images_train = e[train_idxs], digits[train_idxs], y[train_idxs], \
        colors[train_idxs], images[train_idxs]
    e_val, digits_val, y_val, colors_val, images_val = e[val_idxs], digits[val_idxs], y[val_idxs], colors[val_idxs], \
        images[val_idxs]
    assert len(np.unique(y_train)) == len(np.unique(e_train)) == N_CLASSES
    assert len(np.unique(e_train)) == len(np.unique(e_val)) == N_ENVS
    data_train = make_dataloader((e_train, digits_train, y_train, colors_train, images_train), batch_size, n_workers, True)
    data_val = make_dataloader((e_val, digits_val, y_val, colors_val, images_val), batch_size, n_workers, False)
    return data_train, data_val