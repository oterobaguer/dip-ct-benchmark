import argparse

from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters
from dliplib.utils.helper import load_standard_dataset
from dliplib.reconstructors import get_reconstructor
from dliplib.utils.reports import save_results_table
from dliplib.utils.weights import get_weights_path


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--size_part', type=float, default=1.00)
    parser.add_argument('--log_dir', type=str, default=None)
    return parser


def main():
    # load data
    options = get_parser().parse_args()

    dataset = load_standard_dataset(options.dataset, ordered=True)
    ray_trafo = dataset.ray_trafo

    X = ray_trafo.range

    lodopab_cache_file_names = {
        'train': [None, None],
        'validation': [None, None]
    }

    if options.method == 'fbpunet':
        X = ray_trafo.domain
        train_path = 'cache_train_{}_fbp.npy'.format(options.dataset)
        validation_path = 'cache_validation_{}_fbp.npy'.format(options.dataset)
        lodopab_cache_file_names = {
            'train': [train_path, None],
            'validation': [validation_path, None]
        }

    cached_dataset = CachedDataset(dataset,
                                   space=(X, ray_trafo.domain),
                                   cache_files=lodopab_cache_file_names,
                                   size_part=options.size_part)

    test_data = dataset.get_data_pairs('validation',
                                       cached_dataset.validation_len)
    print('validation size: %d' % len(test_data))

    reconstructor = get_reconstructor(options.method,
                                      dataset=options.dataset,
                                      size_part=options.size_part,
                                      pretrained=False)
    print(reconstructor.hyper_params)

    full_name = '{}_{}_{}'.format(
        options.dataset, options.method, options.size_part)
    print(full_name)
    reconstructor.save_best_learned_params_path = get_weights_path(full_name)
    reconstructor.log_dir = options.log_dir
    reconstructor.num_data_loader_workers = 16

    # create a Dival task table and run it
    task_table = TaskTable()
    task_table.append(
        reconstructor=reconstructor,
        measures=[PSNR, SSIM],
        test_data=test_data,
        dataset=cached_dataset,
        hyper_param_choices=[reconstructor.hyper_params]
    )
    results = task_table.run()

    # save report
    save_results_table(results, full_name)


if __name__ == '__main__':
    main()
