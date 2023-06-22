import math
import numpy as np


def softplus(x):
    return np.log(1 + np.exp(x))


def size_to_n_tril(size):
    '''
    Return the number of nonzero entries in a square lower triangular matrix with size rows/columns
    '''
    return int(size * (size + 1) / 2)


def n_tril_to_size(n_tril):
    '''
    Return the number of rows/columns in a square lower triangular matrix with n_tril nonzero entries
    '''
    return int((-1 + math.sqrt(1 + 8 * n_tril)) / 2)


def arr_to_scale_tril(arr):
    '''
    Returns a lower triangular matrix with nonzero diagonal entries
    '''
    n_envs, n_tri = arr.shape
    size = n_tril_to_size(n_tri)
    tril = np.zeros((n_envs, size, size), dtype='float32')
    tril[:, *np.tril_indices(n=size, m=size)] = arr
    diag_idxs = np.arange(size)
    tril[:, diag_idxs, diag_idxs] = softplus(tril[:, diag_idxs, diag_idxs])
    return tril


def arr_to_cov_mat(arr):
    '''
    Cholesky decomposition A = LL^T
    '''
    tril = arr_to_scale_tril(arr)
    return np.matmul(tril, np.transpose(tril, axes=(0, 2, 1)))


def make_raw_data(seed, n_envs, n_examples_per_env, size):
    rng = np.random.RandomState(seed)

    z_c_mu = rng.randn(n_envs, size)
    z_c_cov = arr_to_cov_mat(rng.randn(n_envs, size_to_n_tril(size)))

    z_s_mu = rng.randn(n_envs, size)
    z_s_cov = arr_to_cov_mat(rng.randn(n_envs, size_to_n_tril(size)))

    A = rng.randn(size, size)
    y_cov = 0.1 * np.eye(size)

    half_size = size // 2
    B = rng.randn(size, half_size)

    z_c, y, z_s = [], [], []
    for env_idx in range(n_envs):
        z_c_env = rng.multivariate_normal(z_c_mu[env_idx], z_c_cov[env_idx], n_examples_per_env)
        y_env = z_c_env.dot(A) + rng.multivariate_normal(np.zeros(size), y_cov, n_examples_per_env)
        z_s_env = rng.multivariate_normal(z_s_mu[env_idx], z_s_cov[env_idx], n_examples_per_env)
        z_s_env[:, :half_size] += y_env.dot(B)
        z_c.append(z_c_env)
        y.append(y_env)
        z_s.append(z_s_env)
    z_c, y, z_s = np.row_stack(z_c), np.row_stack(y), np.row_stack(z_s)
    return z_c, y, z_s