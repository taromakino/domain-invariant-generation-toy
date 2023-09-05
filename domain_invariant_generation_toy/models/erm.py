import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.enums import Task
from utils.nn_utils import MLP


class ERM(pl.LightningModule):
    def __init__(self, task, x_size, h_sizes, lr):
        super().__init__()
        self.save_hyperparameters()
        self.task = task
        self.lr = lr
        self.model = MLP(x_size, h_sizes, 1)
        self.val_acc = Accuracy('binary')
        self.test_acc = Accuracy('binary')

    def classify_y_x(self, x, y):
        y_pred = self.model(x)
        log_prob_y_x = -F.binary_cross_entropy_with_logits(y_pred, y)
        return y_pred, log_prob_y_x

    def classify_c_x(self, x, c):
        c_pred = self.model(x)
        log_prob_c_x = -F.binary_cross_entropy_with_logits(c_pred, c)
        return c_pred, log_prob_c_x

    def training_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.ERM_Y_X:
            y_pred, log_prob_y_x  = self.classify_y_x(x, y)
            loss = -log_prob_y_x
            return loss
        elif self.task == Task.ERM_C_X:
            c_pred, log_prob_c_x = self.classify_c_x(x, c)
            loss = -log_prob_c_x
            return loss

    def validation_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.ERM_Y_X:
            y_pred, log_prob_y_x  = self.classify_y_x(x, y)
            loss = -log_prob_y_x
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.val_acc.update(y_pred_class, y.long())
        elif self.task == Task.ERM_C_X:
            c_pred, log_prob_c_x = self.classify_c_x(x, c)
            loss = -log_prob_c_x
            self.log('val_loss', loss, on_step=False, on_epoch=True)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.val_acc.update(c_pred_class, c.long())

    def on_validation_epoch_end(self):
        self.log('val_acc', self.val_acc.compute())

    def test_step(self, batch, batch_idx):
        x, y, e, c, s = batch
        if self.task == Task.ERM_Y_X:
            y_pred, log_prob_y_x  = self.classify_y_x(x, y)
            y_pred_class = (torch.sigmoid(y_pred) > 0.5).long()
            self.test_acc.update(y_pred_class, y.long())
        elif self.task == Task.ERM_C_X:
            c_pred, log_prob_c_x = self.classify_c_x(x, c)
            c_pred_class = (torch.sigmoid(c_pred) > 0.5).long()
            self.test_acc.update(c_pred_class, c.long())

    def on_test_epoch_end(self):
        self.log('test_acc', self.test_acc.compute())

    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)