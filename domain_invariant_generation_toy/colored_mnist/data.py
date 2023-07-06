import numpy as np
import os
import torch
from torchvision import datasets
from utils.nn_utils import make_dataloader


def make_environment(images, labels, e_prob, e_arr):
    def torch_bernoulli(p, size):
        return (torch.rand(size) < p).float()

    def torch_xor(a, b):
        return (a - b).abs()  # Assumes both inputs are either 0 or 1

    # 2x subsample for computational convenience
    images = images.reshape((-1, 28, 28))[:, ::2, ::2]
    # Assign a binary label based on the digit; flip label with probability 0.25
    labels = torch_xor(labels, torch_bernoulli(0.25, len(labels)))
    # Assign a color based on the label; flip the color with probability e
    colors = torch_xor(labels, torch_bernoulli(e_prob, len(labels)))
    # Apply the color to the image by zeroing out the other color channel
    images = torch.stack([images, images], dim=1)
    images[torch.tensor(range(len(images))), (1 - colors).long(), :, :] *= 0
    images = images / 255
    images = images.flatten(start_dim=1)
    e_arr = torch.repeat_interleave(torch.tensor(e_arr, dtype=torch.float32)[None], len(images), dim=0)
    return {
        'x': images,
        'y': labels[:, None],
        'e': e_arr
    }


def make_data(train_ratio, batch_size, n_workers):
    rng = np.random.RandomState(0)
    mnist = datasets.MNIST(os.environ['DATA_DPATH'], train=True, download=True)
    binary_idxs = np.where(mnist.targets <= 1)
    images, binary_digits = mnist.data[binary_idxs], mnist.targets[binary_idxs].float()
    envs = [
        make_environment(images[::2], binary_digits[::2], 0.1, [1, 0, 0]),
        make_environment(images[1::2], binary_digits[1::2], 0.9, [0, 1, 0])
    ]
    x = torch.cat((envs[0]['x'], envs[1]['x']))
    y = torch.cat((envs[0]['y'], envs[1]['y']))
    e = torch.cat((envs[0]['e'], envs[1]['e']))
    n_examples = len(x)
    n_train = int(train_ratio * n_examples)
    train_idxs = rng.choice(n_examples, n_train, replace=False)
    val_idxs = np.setdiff1d(np.arange(n_examples), train_idxs)
    x_train, y_train, e_train = x[train_idxs], y[train_idxs], e[train_idxs]
    x_val, y_val, e_val = x[val_idxs], y[val_idxs], e[val_idxs]
    data_train = make_dataloader((x_train, y_train, e_train), batch_size, n_workers, True)
    data_val = make_dataloader((x_val, y_val, e_val), batch_size, n_workers, False)
    return data_train, data_val