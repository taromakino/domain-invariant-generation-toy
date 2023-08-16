import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.nn_utils import MLP


class ERM(pl.LightningModule):
    def __init__(self, stage, x_size, h_sizes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.stage = stage
        self.lr = lr
        self.model = MLP(x_size, h_sizes, 1, nn.ReLU)
        self.acc = Accuracy('binary')

    def forward(self, x, y, e):
        pred = self.model(x)
        return F.binary_cross_entropy_with_logits(pred, y)

    def training_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('train_loss', loss, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss = self.forward(*batch)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self.model(x)
        y_pred_class = (torch.sigmoid(y_pred) > 0.5).int()
        acc = self.acc(y_pred_class, y)
        self.log(f'{self.stage}_acc', acc, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)