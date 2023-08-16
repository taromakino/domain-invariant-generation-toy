import os
import pytorch_lightning as pl
from argparse import ArgumentParser
from data import MAKE_DATA
from models.erm import ERM
from models.vae import VAE
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from utils.file import save_file


def make_model(args, x_size):
    model_class = ERM if args.is_erm else VAE
    if args.stage == 'train':
        if args.is_erm:
            model = model_class(args.stage, x_size, args.h_sizes, args.lr)
        else:
            model = model_class(args.dpath, args.seed, args.stage, x_size, args.z_size, args.h_sizes,
                args.classifier_mult, args.posterior_reg_mult, args.lr, args.lr_inference, args.n_steps)
    else:
        model = model_class.load_from_checkpoint(args.ckpt_fpath)
    model.stage = args.stage
    return model


def make_trainer(args):
    if args.stage == 'train':
        return pl.Trainer(
            logger=CSVLogger(args.dpath, name='', version=args.seed),
            callbacks=[
                EarlyStopping(monitor='val_loss', patience=int(args.early_stop_ratio * args.n_epochs)),
                ModelCheckpoint(monitor='val_loss', filename='best')],
            max_epochs=args.n_epochs)
    elif args.stage == 'train_q':
        return pl.Trainer(
            logger=CSVLogger(args.dpath, name='', version=args.seed),
            max_epochs=1)
    else:
        return pl.Trainer(
            logger=CSVLogger(args.dpath, name='', version=args.seed),
            max_epochs=1,
            inference_mode=False)


def main(args):
    save_file(args, os.path.join(args.dpath, f'version_{args.seed}', 'args.pkl'))
    pl.seed_everything(args.seed)
    data_train, data_val, data_test = MAKE_DATA[args.dataset](args.train_ratio, args.batch_size)
    x_train, _, _ = data_train.dataset[:]
    model = make_model(args, x_train.shape[1])
    trainer = make_trainer(args)
    if args.stage == 'train':
        trainer.fit(model, data_train, data_val, ckpt_path=args.ckpt_fpath)
    elif args.stage == 'train_q':
        trainer.fit(model, data_train)
    elif args.stage == 'test':
        trainer.test(model, data_test)
    else:
        raise ValueError


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, required=True)
    parser.add_argument('--dpath', type=str, default='results')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--stage', type=str, required=True)
    parser.add_argument('--ckpt_fpath', type=str, default=None)
    parser.add_argument('--train_ratio', type=float, default=0.8)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--is_erm', action='store_true')
    parser.add_argument('--z_size', type=int, default=50)
    parser.add_argument('--h_sizes', nargs='+', type=int, default=[128, 128])
    parser.add_argument('--classifier_mult', type=float, default=1)
    parser.add_argument('--posterior_reg_mult', type=float, default=0.01)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--lr_inference', type=float, default=1e-4)
    parser.add_argument('--n_steps', type=int, default=1000)
    parser.add_argument('--n_epochs', type=int, default=500)
    parser.add_argument("--early_stop_ratio", type=float, default=0.1)
    args = parser.parse_args()
    main(args)