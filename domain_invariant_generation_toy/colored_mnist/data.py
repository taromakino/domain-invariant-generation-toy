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
    zero_idxs = np.where(digits == 0)[0]
    one_idxs = np.where(digits == 1)[0]
    env0_idxs = []
    env0_idxs.append(rng.choice(zero_idxs, int(0.75 * len(zero_idxs)), replace=False))
    env0_idxs.append(rng.choice(one_idxs, int(0.25 * len(one_idxs)), replace=False))
    env0_idxs = np.concatenate(env0_idxs)
    env1_idxs = np.setdiff1d(np.arange(n_total), env0_idxs)
    digits_env0, y_env0, colors_env0, images_env0 = make_environment(rng, images[env0_idxs], digits[env0_idxs], 0.25)
    digits_env1, y_env1, colors_env1, images_env1 = make_environment(rng, images[env1_idxs], digits[env1_idxs], 0.75)
    e = torch.zeros(n_total)[:, None]
    e[env1_idxs] = 1
    digits = torch.cat((digits_env0, digits_env1))[:, None].float()
    y = torch.cat((y_env0, y_env1))[:, None].float()
    colors = torch.cat((colors_env0, colors_env1))[:, None].float()
    images = torch.cat((images_env0, images_env1))
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