import os
import pandas as pd
from argparse import ArgumentParser


def main(args):
    dnames = sorted(os.listdir(args.dpath))
    df = []
    for dname in dnames:
        fpath = os.path.join(args.dpath, dname, f'version_{args.seed}', 'metrics.csv')
        df_elem = pd.read_csv(fpath)
        df_elem.drop(['epoch', 'step'], axis=1, inplace=True)
        df_elem.index = [dname]
        df.append(df_elem)
    df = pd.concat(df)
    df.to_csv(os.path.join(args.dpath, 'summary.csv'))


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--dpath', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    main(parser.parse_args())