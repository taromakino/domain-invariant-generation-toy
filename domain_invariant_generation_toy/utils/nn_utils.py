import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_class):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(activation_class())
            module_list.append(nn.Dropout())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))


def make_dataloader(data_tuple, batch_size, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size)


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
    Return a lower triangular matrix with nonzero diagonal entries
    '''
    batch_size, n_tri = arr.shape
    size = n_tril_to_size(n_tri)
    cov = torch.zeros(batch_size, size, size, dtype=torch.float32, device=arr.device)
    cov[:, *torch.tril_indices(size, size)] = arr
    diag_idxs = torch.arange(size)
    cov[:, diag_idxs, diag_idxs] = F.softplus(cov[:, diag_idxs, diag_idxs])
    return cov


def scale_tril_to_cov(tril):
    '''
    Return a full covariance matrix
    '''
    return torch.bmm(tril, torch.transpose(tril, 1, 2))


def arr_to_cov(arr):
    '''
    Return a full covariance matrix
    '''
    return scale_tril_to_cov(arr_to_scale_tril(arr))