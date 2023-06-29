import numpy as np
from argparse import ArgumentParser
from causallearn.search.ConstraintBased.PC import pc
from cdt.causality.pairwise import ANM, IGCI, RECI
from data import make_data


PAIRWISE_TESTS = {
    'ANM': ANM,
    'IGCI': IGCI,
    'RECI': RECI
}


def jaccard_similarity(lhs, rhs):
    return len(lhs.intersection(rhs)) / len(lhs.union(rhs))


def get_neighbor_names(cg, var_name, var_names, node_names):
    var_to_node_name = dict((var_name, node_names[i]) for i, var_name in enumerate(var_names))
    node_to_var_name = dict((node_name, var_names[i]) for i, node_name in enumerate(node_names))
    neighbors = cg.G.get_adjacent_nodes(cg.G.get_node(var_to_node_name[var_name]))
    return [node_to_var_name[neighbor.name] for neighbor in neighbors]


def main(args):
    pairwise_test = PAIRWISE_TESTS[args.pairwise_test_name]()
    # z_c - y
    ne_ground_truth = set([f'z_c_{i}' for i in range(args.size)] + [f'z_s_{i}' for i in range(args.size // 2)])
    pa_ground_truth = set([f'z_c_{i}' for i in range(args.size)])
    ne_results, pa_results = [], []
    for seed in range(args.n_seeds):
        z_c, y, z_s = make_data(seed, args.n_envs, args.n_examples_per_env, args.size, args.n_components, args.noise_sd)
        var_names = \
            [f'z_c_{i}' for i in range(args.size)] + \
            [f'y_{i}' for i in range(args.size)] + \
            [f'z_s_{i}' for i in range(args.size)]
        data = np.c_[z_c, y, z_s]
        cg = pc(data)
        node_names = cg.G.get_node_names()

        ne_names = []
        for i in range(args.size):
            ne_names += get_neighbor_names(cg, f'y_{i}', var_names, node_names)
        ne_names = [neighbor_name for neighbor_name in ne_names if 'y' not in neighbor_name]
        ne_names = set(ne_names)
        ne_results.append(jaccard_similarity(ne_names, ne_ground_truth))

        pa_names = []
        for neighbor_name in ne_ground_truth:
            neighbor_col_idx = var_names.index(neighbor_name)
            for i in range(args.size):
                y_col_idx = var_names.index(f'y_{i}')
                if pairwise_test.predict_proba((data[:, neighbor_col_idx], data[:, y_col_idx])) > 0:
                    pa_names.append(neighbor_name)
        pa_names = set(pa_names)
        pa_results.append(jaccard_similarity(pa_names, pa_ground_truth))
    print(f'Ne(y): {np.mean(ne_results):.3f} +/- {np.std(ne_results):.3f}')
    print(f'Pa(y): {np.mean(pa_results):.3f} +/- {np.std(pa_results):.3f}')


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--pairwise_test_name', type=str, default='ANM')
    parser.add_argument('--n_seeds', type=int, default=5)
    parser.add_argument('--n_envs', type=int, default=5)
    parser.add_argument('--n_examples_per_env', type=int, default=1000)
    parser.add_argument('--size', type=int, default=10)
    parser.add_argument('--n_components', type=int, default=5)
    parser.add_argument('--noise_sd', type=float, default=0.1)
    args = parser.parse_args()
    assert args.size % 2 == 0
    main(args)