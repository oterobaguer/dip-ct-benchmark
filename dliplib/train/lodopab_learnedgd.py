from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table
from dliplib.utils.weights import get_weights_path
from dliplib.reconstructors.learnedgd import LearnedGDReconstructor


# load data
dataset = load_standard_dataset('lodopab', ordered=True)
ray_trafo = dataset.ray_trafo

global_results = []

sizes = [0.0001, 0.001, 0.01, 0.1, 1.00]
#sizes = [0.0001]
#sizes = [0.001]
#sizes = [0.01]
#sizes = [0.1]

for size_part in sizes:
    cached_dataset = CachedDataset(dataset,
                                   space=(ray_trafo.range, ray_trafo.domain),
                                   cache_files={'train': [None, None],
                                                'validation': [None, None]},
                                   size_part=size_part)

    test_data = dataset.get_data_pairs('validation',
                                       cached_dataset.validation_len)
    print('validation size: %d' % len(test_data))

    full_size_epochs = 10
    # scale number of epochs by 1/size_part, but maximum 1000 times as many
    # epochs as for full size dataset
    epochs = min(500 * full_size_epochs, int(1./size_part * full_size_epochs))

    reconstructor = LearnedGDReconstructor(
        ray_trafo, log_dir='lodopab_learnedgd_{}'.format(size_part),
        save_best_learned_params_path=get_weights_path(
            'lodopab_learnedgd_{}'.format(size_part)))

    # create a Dival task table and run it
    task_table = TaskTable()
    task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                      test_data=test_data, dataset=cached_dataset,
                      hyper_param_choices={'batch_size': [20],
                                           'epochs': [epochs],
                                           'niter': [10],
                                           'lr': [0.00001],
                                            # 'lr_time_decay_rate': [3.2 *
                                                                   # size_part],
                                           'init_frequency_scaling': [0.7],
                                           'init_weight_xavier_normal': [True],
                                           'init_weight_gain': [0.001]})
    results = task_table.run()

    # save report
    save_results_table(results, 'lodopab_learnedgd_{}'.format(size_part))

    # select best parameters and save them
    best_choice, best_error = select_hyper_best_parameters(results)
    params = Params(best_choice)
    params.save('lodopab_learnedgd_{}'.format(size_part))

    # retrain the model with the optimal parameters and save the weights

    # reconstructor = LearnedGDReconstructor(dataset.ray_trafo, hyper_params=params.dict)
    # reconstructor.train(cached_dataset)

#    save_weights(reconstructor.model, 'lodopab_learnedgd_{}'.format(size_part))
