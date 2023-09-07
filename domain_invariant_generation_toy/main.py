import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import MAKE_DATA, X_SIZE
from models.erm import ERM
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task
from utils.file import save_file
from utils.nn_utils import make_dataloader


def make_trainer(task_dpath, seed, n_epochs, early_stop_ratio, is_train):
    if is_train:
        return pl.Trainer(
            logger=CSVLogger(task_dpath, name='', version=seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(early_stop_ratio * n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=n_epochs)
    else:
        return pl.Trainer(
            logger=CSVLogger(task_dpath, name='', version=seed),
            max_epochs=1,
            inference_mode=False)


def main(args):
    pl.seed_everything(args.seed)
    task_dpath = os.path.join(args.dpath, args.task.value)
    save_file(args, os.path.join(task_dpath, f'version_{args.seed}', 'args.pkl'))
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size)
    if args.task == Task.ERM_Y_C or args.task == Task.ERM_Y_S or args.task == Task.ERM_Y_X:
        model = ERM(args.task, X_SIZE[args.dataset], args.h_sizes, args.lr)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, True)
        trainer.fit(model, data_train, data_val)
        trainer.test(model, data_test, ckpt_path='best')
    elif args.task == Task.TRAIN_VAE:
        model = Model(task_dpath, args.seed, args.task, X_SIZE[args.dataset], args.z_size, args.h_sizes,
            args.z_norm_mult, args.lr, args.lr_inference, args.n_steps)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, True)
        trainer.fit(model, data_train, data_val)
    elif args.task == Task.INFER_Z_TRAIN or args.task == Task.INFER_Z_VAL or args.task == Task.INFER_Z_TEST:
        ckpt_fpath = os.path.join(args.dpath, Task.TRAIN_VAE.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')
        model = Model.load_from_checkpoint(ckpt_fpath, dpath=task_dpath, task=args.task, lr_inference=args.lr_inference,
            n_steps=args.n_steps)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, False)
        if args.task == Task.INFER_Z_TRAIN:
            trainer.test(model, data_train)
        elif args.task == Task.INFER_Z_VAL:
            trainer.test(model, data_val)
        else:
            trainer.test(model, data_test)
    else:
        zc_train, y_train = torch.load(os.path.join(args.dpath, Task.INFER_Z_TRAIN.value, f'version_{args.seed}', 'zy.pt'))
        zc_val, y_val = torch.load(os.path.join(args.dpath, Task.INFER_Z_VAL.value, f'version_{args.seed}', 'zy.pt'))
        zc_test, y_test = torch.load(os.path.join(args.dpath, Task.INFER_Z_TEST.value, f'version_{args.seed}', 'zy.pt'))
        infer_data_train = make_dataloader((zc_train, y_train), args.batch_size, True)
        infer_data_val = make_dataloader((zc_val, y_val), args.batch_size, False)
        infer_data_test = make_dataloader((zc_test, y_test), args.batch_size, False)
        ckpt_fpath = os.path.join(args.dpath, Task.TRAIN_VAE.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')
        model = Model.load_from_checkpoint(ckpt_fpath, dpath=task_dpath, task=args.task)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, True)
        trainer.fit(model, infer_data_train, infer_data_val)
        trainer.test(model, infer_data_test, ckpt_path='best')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--is_erm', action='store_true')
    parser.add_argument('--z_size', type=int, default=50)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--z_norm_mult', type=float, default=1e-3)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_inference', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    main(parser.parse_args())
