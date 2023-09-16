import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import MAKE_DATA, X_SIZE
from models.erm import ERM
from models.model import Model
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.enums import Task
from utils.file import save_file


def main(args):
    def make_trainer(task_dpath, n_epochs, inference_mode):
        return pl.Trainer(
            logger=CSVLogger(task_dpath, name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=n_epochs,
            inference_mode=inference_mode)


    def ckpt_fpath(task):
        return os.path.join(args.dpath, task.value, f'version_{args.seed}', 'checkpoints', 'best.ckpt')


    pl.seed_everything(args.seed)
    task_dpath = os.path.join(args.dpath, args.task.value)
    save_file(args, os.path.join(task_dpath, f'version_{args.seed}', 'args.pkl'))
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size_train, args.batch_size_test)
    if args.task == Task.ERM_Y_C or args.task == Task.ERM_Y_S or args.task == Task.ERM_Y_X:
        model = ERM(args.task, X_SIZE[args.dataset], args.h_sizes, args.lr)
        trainer = make_trainer(task_dpath, args.n_epochs, True)
        trainer.fit(model, data_train, data_val)
        trainer.test(model, data_test, ckpt_path='best')
    elif args.task == Task.VAE:
        model = Model(args.task, X_SIZE[args.dataset], args.z_size, args.h_sizes, args.rank, args.reg_mult,
            args.weight_decay, args.lr, args.lr_inference, args.n_steps)
        trainer = make_trainer(task_dpath, args.n_epochs, True)
        trainer.fit(model, data_train, data_val)
    elif args.task == Task.Q_Z:
        model = Model.load_from_checkpoint(ckpt_fpath(Task.VAE), task=args.task)
        trainer = make_trainer(task_dpath, 1, True)
        trainer.test(model, data_train)
        trainer.save_checkpoint(ckpt_fpath(Task.Q_Z))
    elif args.task == Task.CLASSIFY_TRAIN:
        model = Model.load_from_checkpoint(ckpt_fpath(Task.Q_Z), task=args.task)
        trainer = make_trainer(task_dpath, args.n_epochs, False)
        trainer.fit(model, data_train, data_val)
    else:
        assert args.task == Task.CLASSIFY_TEST
        model = Model.load_from_checkpoint(ckpt_fpath(Task.CLASSIFY_TRAIN), task=args.task, reg_mult=args.reg_mult,
            lr_inference=args.lr_inference, n_steps=args.n_steps)
        trainer = make_trainer(task_dpath, 1, False)
        trainer.test(model, data_test)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--task', type=Task, choices=list(Task), required=True)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size_train', type=int, default=128)
    parser.add_argument('--batch_size_test', type=int, default=2048)
    parser.add_argument('--is_erm', action='store_true')
    parser.add_argument('--z_size', type=int, default=100)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[512, 512])
    parser.add_argument('--rank', type=float, default=50)
    parser.add_argument('--reg_mult', type=float, default=1)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_inference', type=float, default=0.01)
    parser.add_argument('--n_steps', type=int, default=5000)
    parser.add_argument('--is_train', action='store_true')
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    main(parser.parse_args())
