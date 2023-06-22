import math
import torch
import torch.nn.functional as F


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
    batch_size, n_tri = arr.shape
    size = n_tril_to_size(n_tri)
    cov = torch.zeros(batch_size, size, size, dtype=torch.float32, device=arr.device)
    cov[:, *torch.tril_indices(size, size)] = arr
    diag_idxs = torch.arange(size)
    cov[:, diag_idxs, diag_idxs] = F.softplus(cov[:, diag_idxs, diag_idxs])
    return cov
