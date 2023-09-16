import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(nn.LeakyReLU())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))


def make_dataloader(data_tuple, batch_size, is_train):
    return DataLoader(TensorDataset(*data_tuple), shuffle=is_train, batch_size=batch_size)


def arr_to_cov(arr):
    batch_size, size = arr.shape
    cov =  torch.bmm(arr.unsqueeze(2), arr.unsqueeze(1))
    diag_idxs = torch.arange(size)
    cov[:, diag_idxs, diag_idxs] = torch.exp(cov[:, diag_idxs, diag_idxs])
    return cov


def arr_to_tril(arr):
    return torch.linalg.cholesky(arr_to_cov(arr))