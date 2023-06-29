import numpy as np
import torch
import torch.nn as nn
from sklearn.mixture import GaussianMixture
from utils.stats import softplus, sample_isotropic_mvn


def sample_gmm(rng, n_examples, n_components, size):
    dist = GaussianMixture(n_components, covariance_type='diag')
    dist.means_ = 2 * rng.randn(n_components, size)
    dist.covariances_ = softplus(rng.randn(n_components, size))
    dist.weights_ = abs(rng.randn(n_components))
    dist.weights_ = dist.weights_ / sum(dist.weights_)
    return dist.sample(n_examples)[0]


def make_mlp(size, h_size=20):
    return nn.Sequential(
        nn.Linear(size, h_size),
        nn.ReLU(),
        nn.Linear(h_size, size)
    )


def forward(mlp, x):
    return mlp(torch.tensor(x, dtype=torch.float32)).detach().numpy()


def make_data(seed, n_envs, n_examples_per_env, size, n_components, noise_sd):
    rng = np.random.RandomState(seed)
    half_size = size // 2

    zs_to_y = make_mlp(size)
    y_to_zc = make_mlp(half_size)

    zs, y, zc = [], [], []
    for _ in range(n_envs):
        zs_env = sample_gmm(rng, n_examples_per_env, n_components, size)
        y_env = forward(zs_to_y, zs_env + sample_isotropic_mvn(rng, n_examples_per_env, size, noise_sd))
        zc_env = sample_gmm(rng, n_examples_per_env, n_components, size)
        zc_env[:, :half_size] += forward(y_to_zc, y_env[:, :half_size] + zc_env[:, :half_size] +
            sample_isotropic_mvn(rng, n_examples_per_env, half_size, noise_sd))
        zs.append(zs_env)
        y.append(y_env)
        zc.append(zc_env)
    zs = np.vstack(zs)
    y = np.vstack(y)
    zc = np.vstack(zc)
    return zs, y, zc