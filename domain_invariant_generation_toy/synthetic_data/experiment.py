import numpy as np
import pytorch_lightning as pl
from argparse import ArgumentParser
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import minmax_scale
from synthetic_data.data import make_data


def mse(features, target):
    features = features ** 2
    model = LinearRegression()
    model.fit(features, target)
    pred = model.predict(features)
    return mean_squared_error(target, pred)


def jaccard_similarity(lhs, rhs):
    return len(lhs.intersection(rhs)) / len(lhs.union(rhs))


def main(args):
    ground_truth = set([i for i in range(args.size)])
    results = []
    for seed in range(args.n_seeds):
        pl.seed_everything(seed)
        e, zc, y, zs = make_data(seed, args.n_envs, args.n_examples_per_env, args.size, args.n_components, args.noise_sd)
        z = np.hstack((zc, zs))
        z_size = z.shape[1]

        z = minmax_scale(z)
        y = minmax_scale(y)

        cause_idxs = []
        for idx in range(z_size):
            z_i = z[:, idx][:, None]
            # y <- z_i
            target = y
            features = np.hstack((z_i, e))
            # features = z_i
            lhs = mse(features, target)

            # z_i <- y
            target = z_i
            features = np.hstack((y, e))
            # features = y
            rhs = mse(features, target)

            is_zi_cause = lhs < rhs
            if is_zi_cause:
                cause_idxs.append(idx)
        results.append(jaccard_similarity(set(cause_idxs), ground_truth))
    print(f'{np.mean(results):.3f} +/- {np.std(results):.3f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--n_seeds', type=int, default=10)
    parser.add_argument('--n_envs', type=int, default=5)
    parser.add_argument('--n_examples_per_env', type=int, default=10000)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--noise_sd', type=float, default=0.1)
    args = parser.parse_args()
    assert args.size % 2 == 0
    main(args)