import models.erm as erm
import models.vae as vae
import os
import pytorch_lightning as pl
import torch
from argparse import ArgumentParser
from data import MAKE_DATA, X_SIZE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task, EvalStage
from utils.file import save_file
from utils.nn_utils import make_dataloader


def make_data(args):
    if args.task in [
        Task.ERM_ZC,
        Task.ERM_ZS
    ]:
        data_train = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, EvalStage.TRAIN.value,
            f'version_{args.seed}', 'infer.pt')), args.batch_size, True)
        data_val = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, EvalStage.VAL.value,
            f'version_{args.seed}', 'infer.pt')), args.batch_size, False)
        data_test = make_dataloader(torch.load(os.path.join(args.dpath, Task.INFER_Z.value, EvalStage.TEST.value,
            f'version_{args.seed}', 'infer.pt')), args.batch_size, False)
    else:
        data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size)
    if args.eval_stage == EvalStage.TRAIN:
        data_eval = data_train
    elif args.eval_stage == EvalStage.VAL:
        data_eval = data_val
    else:
        assert args.eval_stage == EvalStage.TEST
        data_eval = data_test
    return data_train, data_val, data_eval


def ckpt_fpath(args, task):
    return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


def make_model(args):
    if args.task == Task.ERM_X:
        if args.is_train:
            return erm.ERM_X(X_SIZE[args.dataset], args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_X.load_from_checkpoint(ckpt_fpath(args, args.task))
    elif args.task == Task.ERM_ZC:
        if args.is_train:
            return erm.ERM_ZC(args.z_size, args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_ZC.load_from_checkpoint(ckpt_fpath(args, args.task))
    elif args.task == Task.ERM_ZS:
        if args.is_train:
            return erm.ERM_ZS(args.z_size, args.h_sizes, args.lr, args.weight_decay)
        else:
            return erm.ERM_ZS.load_from_checkpoint(ckpt_fpath(args, args.task))
    elif args.task == Task.VAE:
        return vae.VAE(args.task, X_SIZE[args.dataset], args.z_size, args.rank, args.h_sizes, args.beta, args.lr,
            args.reg_mult, args.weight_decay, args.alpha, args.lr_infer, args.n_infer_steps)
    elif args.task == Task.Q_Z:
        return vae.VAE.load_from_checkpoint(ckpt_fpath(args, Task.VAE), task=args.task)
    else:
        assert args.task == Task.INFER_Z
        return vae.VAE.load_from_checkpoint(ckpt_fpath(args, Task.Q_Z), task=args.task, alpha=args.alpha,
            lr_infer=args.lr_infer, n_infer_steps=args.n_infer_steps)


def main(args):
    pl.seed_everything(args.seed)
    data_train, data_val_iid, data_eval = make_data(args)
    model = make_model(args)
    if args.task in [
        Task.ERM_X,
        Task.ERM_ZC,
        Task.ERM_ZS
    ]:
        if args.is_train:
            trainer = pl.Trainer(
                logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
                callbacks=[
                    EarlyStopping(monitor='val_metric', mode='max', patience=int(args.early_stop_ratio * args.n_epochs)),
                    ModelCheckpoint(monitor='val_metric', mode='max', filename='best')],
                max_epochs=args.n_epochs)
            trainer.fit(model, data_train, data_val_iid)
        else:
            trainer = pl.Trainer(logger=CSVLogger(os.path.join(args.dpath, args.task.value, args.eval_stage.value),
                name='', version=args.seed), max_epochs=1)
            trainer.test(model, data_eval)
    elif args.task == Task.VAE:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=args.n_epochs)
        trainer.fit(model, data_train, data_val_iid)
        save_file(args, os.path.join(args.dpath, args.task.value, f'version_{args.seed}', 'args.pkl'))
    elif args.task == Task.Q_Z:
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value), name='',
                version=args.seed),
            max_epochs=1)
        trainer.test(model, data_train)
        trainer.save_checkpoint(ckpt_fpath(args, args.task))
        save_file(args, os.path.join(args.dpath, args.task.value, f'version_{args.seed}', 'args.pkl'))
    else:
        assert args.task == Task.INFER_Z
        trainer = pl.Trainer(
            logger=CSVLogger(os.path.join(args.dpath, args.task.value, args.eval_stage.value), name='',
                version=args.seed),
            max_epochs=1,
            inference_mode=False)
        trainer.test(model, data_eval)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--eval_stage', type=EvalStage, choices=list(EvalStage), default='test')
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--z_size', type=int, default=100)
    parser.add_argument('--rank', type=int, default=50)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--beta', type=float, default=1)
    parser.add_argument('--reg_mult', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--alpha', type=float, default=1)
    parser.add_argument('--lr_infer', type=float, default=0.01)
    parser.add_argument('--n_infer_steps', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=200)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    main(parser.parse_args())
