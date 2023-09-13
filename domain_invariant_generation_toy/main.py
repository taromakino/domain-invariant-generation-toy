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


def make_trainer(task_dpath, seed, n_epochs, early_stop_ratio, inference_mode):
    return pl.Trainer(
        logger=CSVLogger(task_dpath, name='', version=seed),
        callbacks=[
            EarlyStopping(monitor='val_loss', patience=int(early_stop_ratio * n_epochs)),
            ModelCheckpoint(monitor='val_loss', filename='best')],
        max_epochs=n_epochs,
        inference_mode=inference_mode)


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
    elif args.task == Task.VAE:
        model = Model(task_dpath, args.seed, args.task, X_SIZE[args.dataset], args.z_size, args.h_sizes, args.weight_decay,
            args.lr, args.lr_inference, args.n_steps)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, True)
        trainer.fit(model, data_train, data_val)
    elif args.task == Task.Q_Z:
        ckpt_fpath = os.path.join(args.dpath, Task.VAE.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')
        model = Model.load_from_checkpoint(ckpt_fpath, task=args.task)
        trainer = make_trainer(task_dpath, args.seed, 1, args.early_stop_ratio, True)
        trainer.test(model, data_train)
        trainer.save_checkpoint(os.path.join(task_dpath, f'version_{args.seed}', 'checkpoints', 'best.ckpt'))
    elif args.task == Task.INFER_Z:
        ckpt_fpath = os.path.join(args.dpath, Task.Q_Z.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')
        model = Model.load_from_checkpoint(ckpt_fpath, dpath=task_dpath, task=args.task, lr_inference=args.lr_inference,
            n_steps=args.n_steps)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, False)
        trainer.test(model, data_test)
    else:
        assert args.task == Task.CLASSIFY
        data_test = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, f'version_{args.seed}',
            'z.pt')), args.batch_size, False)
        ckpt_fpath = os.path.join(args.dpath, Task.Q_Z.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')
        model = Model.load_from_checkpoint(ckpt_fpath, task=args.task)
        trainer = make_trainer(task_dpath, args.seed, args.n_epochs, args.early_stop_ratio, True)
        trainer.fit(model, data_train, data_val)
        trainer.test(model, data_test)


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
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_inference', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    main(parser.parse_args())
