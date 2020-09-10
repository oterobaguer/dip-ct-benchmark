import argparse
from dival import DataPairs

from dliplib.reconstructors import (tvadam_reconstructor,
                                    fbp_reconstructor,
                                    diptv_reconstructor,
                                    learnedgd_reconstructor,
                                    learnedpd_reconstructor,
                                    iradonmap_reconstructor,
                                    fbpunet_reconstructor)

from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests
from dliplib.utils.helper import set_use_latex

set_use_latex()


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--cmap', type=str, default='bone')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=1)
    return parser


def main():
    options = get_parser().parse_args()
    # load data
    dataset = load_standard_dataset(options.dataset)

    test_data = dataset.get_data_pairs('test', 100)
    test_data = list(test_data)[options.start: options.start + options.count]

    obs = list([item[0] for item in test_data])
    gt = list([item[1] for item in test_data])

    test_data = DataPairs(obs, gt, name='test')

    # load reconstructor
    fbp = fbp_reconstructor(options.dataset)
    tv = tvadam_reconstructor(options.dataset, name='TV')
    diptv = diptv_reconstructor(options.dataset)
    learnedgd = learnedgd_reconstructor(options.dataset, size_part=1.0)
    fbpunet = fbpunet_reconstructor(options.dataset, size_part=1.0)
    iradonmap = iradonmap_reconstructor(options.dataset, size_part=1.0)
    learnedpd = learnedpd_reconstructor(options.dataset, size_part=1.0)

    # compute and plot reconstructions
    plot_reconstructors_tests([fbp,
                               tv,
                               diptv,
                               iradonmap,
                               fbpunet,
                               learnedgd,
                               learnedpd],
                              ray_trafo=dataset.ray_trafo,
                              test_data=test_data,
                              save_name='{}-all-{}'.format(
        options.dataset, options.start),
        fig_size=(9, 6.5),
        cmap=options.cmap)


if __name__ == '__main__':
    main()
