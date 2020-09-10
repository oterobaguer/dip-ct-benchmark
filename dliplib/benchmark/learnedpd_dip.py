import argparse
from dival import DataPairs

from dliplib.reconstructors import (learnedpd_reconstructor,
                                    diptv_reconstructor,
                                    tvadam_reconstructor,
                                    learnedpd_dip_reconstructor)
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='ellipses')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--cmap', type=str, default='bone')
    parser.add_argument('--size_part', type=float, default=1.0)
    parser.add_argument('--count', type=int, default=1)
    return parser


def main():
    options = get_parser().parse_args()
    # load data
    dataset = load_standard_dataset(options.dataset)
    test_data = dataset.get_data_pairs('test', 3000)

    # index = [0, 2, 68]
    index = range(options.start, options.start + options.count)
    obs = [test_data[i][0] for i in index]
    gt = [test_data[i][1] for i in index]
    test_data = DataPairs(obs, gt, name='test')

    data_size = options.size_part

    # load reconstructors
    diptv = diptv_reconstructor(options.dataset)
    learnedpd = learnedpd_reconstructor(options.dataset, data_size)
    learnedpd_dip = learnedpd_dip_reconstructor(options.dataset, data_size)

    # compute example reconstructions
    plot_reconstructors_tests([diptv, learnedpd, learnedpd_dip],
                              dataset.ray_trafo,
                              test_data,
                              save_name='{}-learnedpd-dip-{}-{}'.format(
                                  options.dataset, data_size, options.start),
                              fig_size=(9, 3),
                              cmap=options.cmap)


if __name__ == "__main__":
    main()
