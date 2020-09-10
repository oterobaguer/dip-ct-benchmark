import argparse

from dival import TaskTable, DataPairs
from dival.measure import SSIM, PSNR

from dliplib.reconstructors import get_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.reports import save_results_table


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--method', type=str)
    parser.add_argument('--size_part', type=float, default=None)
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--count', type=int, default=None)
    return parser


def main():
    """Main function"""
    options = get_parser().parse_args()
    # options.dataset = 'lodopab' | 'ellipses' | 'lodopab-sparse'

    dataset = load_standard_dataset(options.dataset)
    test_data = dataset.get_data_pairs('test', 100)

    obs = list(y for y, x in test_data)
    gt = list(x for y, x in test_data)
    start = options.start
    count = options.count
    if count is None:
        count = len(test_data)
    test_data = DataPairs(obs[start: start + count],
                          gt[start: start + count], name='test')

    # load reconstructor
    reconstructor = get_reconstructor(
        options.method, options.dataset, options.size_part)

    # eval on the test-set
    print('Reconstructor: %s' % options.method)
    print('Dataset: %s' % options.dataset)
    print('Offset: %d' % start)
    print('Count: %d' % count)

    task_table = TaskTable()
    task_table.append(
        reconstructor=reconstructor,
        measures=[PSNR, SSIM],
        test_data=test_data,
        options={'skip_training': True}
    )
    task_table.run()

    print(task_table.results.to_string(show_columns=['misc']))

    if options.size_part is not None:
        save_path = '{}_{}_{}_eval'.format(
            options.dataset, options.method, options.size_part)
    else:
        save_path = '{}_{}_eval'.format(options.dataset, options.method)
    save_path += '_offset_%d' % start

    save_results_table(task_table.results, save_path)


if __name__ == '__main__':
    main()
