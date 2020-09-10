import argparse
from dival import DataPairs

from dliplib.reconstructors import tv_reconstructor
from dliplib.reconstructors import tvadam_reconstructor
from dliplib.reconstructors import fbp_reconstructor

from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


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
    tv = tv_reconstructor(options.dataset, name='TV')
    tvadam = tvadam_reconstructor(options.dataset, name='TV-Adam')

    # compute and plot reconstructions
    plot_reconstructors_tests([fbp, tv, tvadam],
                              ray_trafo=dataset.ray_trafo,
                              test_data=test_data,
                              save_name='{}-tv-tvadam-{}'.format(
        options.dataset, options.start),
        fig_size=(9, 3),
        cmap=options.cmap)


if __name__ == '__main__':
    main()
