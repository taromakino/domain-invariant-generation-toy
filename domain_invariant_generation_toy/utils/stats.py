import numpy as np


def softplus(x):
    return np.log(1 + np.exp(x))


def sample_isotropic_mvn(rng, n_examples, size, sd):
    return rng.multivariate_normal(np.zeros(size), (sd ** 2) * np.eye(size), n_examples)