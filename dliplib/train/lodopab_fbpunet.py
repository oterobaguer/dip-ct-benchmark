from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.weights import save_weights, get_weights_path
from dliplib.utils.reports import save_results_table
from dliplib.reconstructors.fbpunet import FBPUNetReconstructor


lodopab_cache_file_names = {
    'train': ['cache_train_lodopab_fbp.npy', None],
    'validation': ['cache_validation_lodopab_fbp.npy', None]
}

# load data
dataset = load_standard_dataset('lodopab', ordered=True)
ray_trafo = dataset.ray_trafo

sizes = [0.0001, 0.001, 0.01, 0.1, 1.00]
#sizes = [0.0001]
#sizes = [0.001]
#sizes = [0.01]
#sizes = [0.1]


for size_part in sizes:
    cached_dataset = CachedDataset(dataset,
                                   space=(ray_trafo.domain, ray_trafo.domain),
                                   cache_files=lodopab_cache_file_names,
                                   size_part=size_part)

    # create a Dival task table and run it
    task_table = TaskTable()
    test_data = dataset.get_data_pairs('validation',
                                       cached_dataset.validation_len)
    print('validation size: %d' % len(test_data))

    full_size_epochs = 250
    # scale number of epochs by 1/size_part, but maximum 100 times as many
    # epochs as for full size dataset
    epochs = min(100 * full_size_epochs, int(1./size_part * full_size_epochs))

    reconstructor = FBPUNetReconstructor(ray_trafo,
        log_dir='lodopab_fbpunet_{}'.format(size_part),
        save_best_learned_params_path=get_weights_path(
            'lodopab_fbpunet_{}'.format(size_part)))


    task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                      test_data=test_data, dataset=cached_dataset,
                      hyper_param_choices={'scales': [5],
                                           'skip_channels': [4],
                                           'batch_size': [32],
                                           'epochs': [epochs],
                                           'lr': [0.01],
                                           'filter_type': ['Hann'],
                                           'frequency_scaling': [1.0]
                                           })
    results = task_table.run()

    # save report
    save_results_table(results, 'lodopab_fbpunet_{}'.format(size_part))

    # select best parameters and save them
    best_choice, best_error = select_hyper_best_parameters(results)
    params = Params(best_choice)
    params.save('lodopab_fbpunet_{}'.format(size_part))
