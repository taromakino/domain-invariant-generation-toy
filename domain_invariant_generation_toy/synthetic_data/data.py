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


def make_mlp(input_size, output_size, h_size=20):
    return nn.Sequential(
        nn.Linear(input_size, h_size),
        nn.ReLU(),
        nn.Linear(h_size, output_size)
    )


def forward(mlp, x):
    return mlp(torch.tensor(x, dtype=torch.float32)).detach().numpy()


def make_data(seed, n_envs, n_examples_per_env, size, n_components, noise_sd):
    rng = np.random.RandomState(seed)
    half_size = size // 2

    zc_to_y = make_mlp(size, 1)
    y_to_zs = make_mlp(half_size, half_size)

    e, zc, y, zs = [], [], [], []
    for env_idx in range(n_envs):
        e_env = np.zeros((n_examples_per_env, n_envs))
        e_env[:, env_idx] = 1
        e.append(e_env)
        zc_env = sample_gmm(rng, n_examples_per_env, n_components, size)
        zc.append(zc_env)
        y_env = forward(zc_to_y, zc_env + sample_isotropic_mvn(rng, n_examples_per_env, size, noise_sd))
        y.append(y_env)
        zs_env = sample_gmm(rng, n_examples_per_env, n_components, size)
        zs_env[:, :half_size] += forward(y_to_zs, y_env + zs_env[:, :half_size] +
            sample_isotropic_mvn(rng, n_examples_per_env, half_size, noise_sd))
        zs.append(zs_env)
    e = np.vstack(e)
    zc = np.vstack(zc)
    y = np.vstack(y)
    zs = np.vstack(zs)
    return e, zc, y, zs