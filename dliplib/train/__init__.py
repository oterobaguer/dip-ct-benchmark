from dliplib.utils.data import generate_dataset_cache
from dliplib.utils.helper import load_standard_dataset


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

lodopab_200_cache_file_names = {
    'train': ['cache_train_lodopab_200_fbp.npy'],
    'validation': ['cache_validation_lodopab_200_fbp.npy']
}


def generate_cache(dataset_name, file_names, frequency_scaling=1.0, only_fbp=False):
    dataset = load_standard_dataset(dataset_name, ordered=False)
    ray_trafo = dataset.ray_trafo
    for part in ['train', 'validation']:
        generate_dataset_cache(
            dataset=dataset,
            part=part,
            ray_trafo=ray_trafo,
            file_names=file_names[part],
            frequency_scaling=frequency_scaling,
            filter_type='Hann',
            only_fbp=only_fbp
        )


if __name__ == '__main__':
    generate_cache(
        dataset_name='ellipses',
        file_names=ellipses_cache_file_names,
        only_fbp=True,
        frequency_scaling=1.0
 
    )
    generate_cache(
        dataset_name='lodopab',
        file_names=lodopab_cache_file_names,
        only_fbp=True,
        frequency_scaling=1.0
    )
    generate_cache(
        dataset_name='lodopab_200',
        file_names=lodopab_200_cache_file_names,
        only_fbp=True,
        frequency_scaling=1.0
    )
