import argparse

from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table
from dliplib.utils.weights import get_weights_path
from dliplib.reconstructors.learnedpd import LearnedPDReconstructor


def get_parser():
    """Adds arguments to the command"""
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str)
    parser.add_argument('--size_part', type=float, default=1.00)
    return parser


def main():
    # load data
    options = get_parser().parse_args()

    dataset = load_standard_dataset('lodopab', ordered=True)
    ray_trafo = dataset.ray_trafo

    global_results = []

    sizes = [0.001, 0.01, 0.1, 1.00]
    #sizes = [0.0001]
    #sizes = [0.001]
    #sizes = [0.01]
    #sizes = [0.1]

    for size_part in sizes:
        cached_dataset = CachedDataset(dataset,
                                       space=(ray_trafo.range,
                                              ray_trafo.domain),
                                       cache_files={'train': [None, None],
                                                    'validation': [None, None]},
                                       size_part=size_part)

        test_data = dataset.get_data_pairs('validation',
                                           cached_dataset.validation_len)
        print('validation size: %d' % len(test_data))

        full_size_epochs = 10 if size_part >= 0.10 else 5
        lr = 0.0001 if size_part >= 0.10 else 0.001
        # scale number of epochs by 1/size_part, but maximum 1000 times as many
        # epochs as for full size dataset
        epochs = min(1000 * full_size_epochs,
                     int(1./size_part * full_size_epochs))

        reconstructor = LearnedPDReconstructor(
            ray_trafo,
            log_dir='lodopab_learnedpd_{}'.format(size_part),
            save_best_learned_params_path=get_weights_path('lodopab_learnedpd_{}'.format(size_part)))

        # create a Dival task table and run it
        task_table = TaskTable()
        task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                          test_data=test_data, dataset=cached_dataset,
                          hyper_param_choices={'batch_size': [1],
                                               'epochs': [epochs],
                                               'niter': [10],
                                               'internal_ch': [64 if
                                                               size_part >= 0.10
                                                               else 32],
                                               'lr': [lr],
                                               'lr_min': [lr],
                                               'init_fbp': [True],
                                               'init_frequency_scaling': [0.7]})
        results = task_table.run()

        # save report
        save_results_table(results, 'lodopab_learnedpd_{}'.format(size_part))

        # select best parameters and save them
        best_choice, best_error = select_hyper_best_parameters(results)
        params = Params(best_choice)
        params.save('lodopab_learnedpd_{}'.format(size_part))

        # retrain the model with the optimal parameters and save the weights

        # reconstructor = LearnedPDReconstructor(dataset.ray_trafo, hyper_params=params.dict)
        # reconstructor.train(cached_dataset)

    #    save_weights(reconstructor.model, 'lodopab_learnedpd_{}'.format(size_part))
