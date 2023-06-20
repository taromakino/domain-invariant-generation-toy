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


def jaccard_similarity(lhs, rhs):
    return len(lhs.intersection(rhs)) / len(lhs.union(rhs))