import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger


class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, output_dim, activation_class):
        super().__init__()
        module_list = []
        last_in_dim = input_dim
        for hidden_dim in hidden_dims:
            module_list.append(nn.Linear(last_in_dim, hidden_dim))
            module_list.append(activation_class())
            last_in_dim = hidden_dim
        module_list.append(nn.Linear(last_in_dim, output_dim))
        self.module_list = nn.Sequential(*module_list)

    def forward(self, *args):
        return self.module_list(torch.hstack(args))


def make_trainer(dpath, seed, n_epochs, early_stop_ratio):
    return pl.Trainer(
        logger=CSVLogger(dpath, name='', version=seed),
        callbacks=[
            EarlyStopping(monitor="val_loss", patience=int(early_stop_ratio * n_epochs)),
            ModelCheckpoint(monitor="val_loss", filename="best")],
        max_epochs=n_epochs)