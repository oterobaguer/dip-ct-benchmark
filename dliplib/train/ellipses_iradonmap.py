from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table
from dliplib.utils.weights import save_weights, get_weights_path
from dliplib.reconstructors.iradonmap import IRadonMapReconstructor

ellipses_cache_file_names = {
    'train': ['cache_train_ellipses_obs.npy',
              'cache_train_ellipses_gts.npy'],
    'validation': ['cache_validation_ellipses_obs.npy',
                   'cache_validation_ellipses_gts.npy']
}

# load data
dataset = load_standard_dataset('ellipses')
ray_trafo = dataset.ray_trafo

global_results = []

full_size_epochs = 100
sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

# Divide training into several GPUs:
# sizes = [0.005, 0.01, 0.50]
# sizes = [0.02, 0.05, 0.10, 0.25]
# sizes = [0.001, 0.002, 1.00]

for size_part in sizes:
    cached_dataset = CachedDataset(dataset=dataset,
                                   space=(ray_trafo.range, ray_trafo.domain),
                                   cache_files=ellipses_cache_file_names,
                                   size_part=size_part)

    test_data = dataset.get_data_pairs('validation',
                                       cached_dataset.validation_len)
    print('validation size: %d' % len(test_data))

    reconstructor = IRadonMapReconstructor(
                ray_trafo=ray_trafo,
                log_dir='ellipses_iradonmap/' + str(size_part),
                save_best_learned_params_path=get_weights_path(
                    'ellipses_iradonmap_{}'.format(size_part)))

    epochs = min(10 * full_size_epochs, int(1./size_part * full_size_epochs))

    # create a Dival task table and run it
    task_table = TaskTable()
    task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                      test_data=test_data, dataset=cached_dataset,
                      hyper_param_choices={'scales': [5],
                                           'skip_channels': [4],
                                           'batch_size': [64],
                                           'epochs': [epochs],
                                           'fully_learned': [True],
                                           'lr': [0.01],
                                           'use_sigmoid': [True]})
    results = task_table.run()

    # save report
    save_results_table(results, 'ellipses_iradonmap_{}'.format(size_part))

    # select best parameters and save them
    best_choice, best_error = select_hyper_best_parameters(results)
    params = Params(best_choice)
    params.save('ellipses_iradonmap_{}'.format(size_part))

    save_weights(reconstructor, 'ellipses_iradonmap_{}'.format(size_part))
    global_results.append(best_error)

print(global_results)
