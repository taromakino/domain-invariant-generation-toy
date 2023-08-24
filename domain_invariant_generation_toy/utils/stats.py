import numpy as np
import torch
import torch.distributions as D


def min_max_scale(x):
    return (x - x.min()) / (x.max() - x.min())


def softplus(x):
    return np.log(1 + np.exp(x))


def sample_isotropic_mvn(rng, n_examples, size, sd):
    return rng.multivariate_normal(np.zeros(size), (sd ** 2) * np.eye(size), n_examples)