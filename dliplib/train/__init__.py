from dival import get_standard_dataset
from dliplib.utils.data import generate_dataset_cache


ellipses_cache_file_names = {
    'train': ['cache_train_ellipses_gts.npy',
              'cache_train_ellipses_obs.npy',
              'cache_train_ellipses_fbp.npy'],
    'validation': ['cache_validation_ellipses_gts.npy',
                   'cache_validation_ellipses_obs.npy',
                   'cache_validation_ellipses_fbp.npy']
}

lodopab_cache_file_names = {
    'train': ['cache_train_lodopab_fbp.npy'],
    'validation': ['cache_validation_lodopab_fbp.npy']
}


def generate_ellipses_cache():
    dataset = get_standard_dataset('ellipses', fixed_seeds=True)
    ray_trafo = dataset.ray_trafo
    for part in ['train', 'validation']:
        generate_dataset_cache(dataset, part=part, ray_trafo=ray_trafo, file_names=ellipses_cache_file_names[part],
                               frequency_scaling=1.0, filter_type='Hann')


def generate_lodopab_cache():
    dataset = get_standard_dataset('lodopab')
    ray_trafo = dataset.ray_trafo
    for part in ['train', 'validation']:
        generate_dataset_cache(dataset, part=part, ray_trafo=ray_trafo, file_names=lodopab_cache_file_names[part],
                               frequency_scaling=1.0, filter_type='Hann', only_fbp=True)


if __name__ == '__main__':
    generate_ellipses_cache()
    # generate_lodopab_cache()