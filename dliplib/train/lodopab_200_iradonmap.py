from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table
from dliplib.utils.weights import get_weights_path
from dliplib.reconstructors.iradonmap import IRadonMapReconstructor


# load data
dataset = load_standard_dataset('lodopab_200', ordered=True)
ray_trafo = dataset.ray_trafo

global_results = []
reconstructor = None
task_table = None

full_size_epochs = 30
sizes = [0.0001, 0.001, 0.01, 0.1, 1.00]
#sizes = [0.0001]
#sizes = [0.001]
#sizes = [0.01]
#sizes = [0.1]
#sizes = [1.00]


for size_part in sizes:
    del(task_table)
    del(reconstructor)

    cached_dataset = CachedDataset(dataset,
                                   space=(ray_trafo.range, ray_trafo.domain),
                                   cache_files={'train': [None, None],
                                                'validation': [None, None]},
                                   size_part=size_part)

    test_data = dataset.get_data_pairs('validation',
                                       cached_dataset.validation_len)
    print('validation size: %d' % len(test_data))

    reconstructor = IRadonMapReconstructor(
                ray_trafo=ray_trafo,
                log_dir='lodopab_200_iradonmap/' + str(size_part),
                save_best_learned_params_path=get_weights_path(
                    'lodopab_200_iradonmap_{}'.format(size_part)))

    epochs = min(10 * full_size_epochs, int(1./size_part * full_size_epochs))

    # create a Dival task table and run it
    task_table = TaskTable()
    task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                      test_data=test_data, dataset=cached_dataset,
                      hyper_param_choices={'scales': [5],
                                           'skip_channels': [4],
                                           'batch_size': [32],
                                           'epochs': [epochs],
                                           'fully_learned': [True],
                                           'lr': [0.01],
                                           'use_sigmoid': [False, True]})
    results = task_table.run()

    # save report
    save_results_table(results, 'lodopab_200_iradonmap_{}'.format(size_part))

    # select best parameters and save them
    best_choice, best_error = select_hyper_best_parameters(results)
    params = Params(best_choice)
    params.save('lodopab_200_iradonmap_{}'.format(size_part))
