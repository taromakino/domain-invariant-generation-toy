import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from argparse import ArgumentParser
from data import MAKE_DATA
from torch.optim import Adam
from torchmetrics import Accuracy
from utils.nn_utils import MLP, make_trainer


class MLPClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dims, lr):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.model = MLP(input_dim, hidden_dims, 1, nn.ReLU)
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
        self.log('test_acc', acc, on_step=False, on_epoch=True)


    def configure_optimizers(self):
        return Adam(self.parameters(), lr=self.lr)


def main(args):
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size)
    x_train, y_train, e_train = data_train.dataset[:]
    model = MLPClassifier(x_train.shape[1], args.h_sizes, args.lr)
    trainer = make_trainer(args.dpath, args.seed, args.n_epochs, args.early_stop_ratio)
    trainer.fit(model, data_train, data_val)
    trainer.test(model, data_test, ckpt_path='best')



if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument('--early_stop_ratio', type=float, default=0.1)
    main(parser.parse_args())