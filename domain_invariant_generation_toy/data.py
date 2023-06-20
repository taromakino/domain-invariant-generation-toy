import numpy as np
from utils.stats import arr_to_cov_mat, size_to_n_tril


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