import numpy as np

from tqdm import tqdm
from odl.tomo import fbp_op
from dliplib.utils.data.datasets import CachedDataset


def generate_dataset_cache(dataset, part, file_names, ray_trafo, filter_type='Hann', frequency_scaling=1.0, size=None, only_fbp=False):
    """
    Write data-paris for a CT dataset part to file.

    Parameters
    ----------
    dataset : :class:`.Dataset`
        CT dataset from which the observations are used.
    part : {``'train'``, ``'validation'``, ``'test'``}
        The data part.
    file_name : str
        The filenames to store the cache at (ending ``.npy``).
    ray_trafo : :class:`odl.tomo.RayTransform`
        Ray transform from which the FBP operator is constructed.
    size : int, optional
        Number of samples to use from the dataset.
        By default, all samples are used.
    """
    fbp = fbp_op(ray_trafo, filter_type=filter_type, frequency_scaling=frequency_scaling)
    num_samples = dataset.get_len(part=part) if size is None else size

    if not only_fbp:
        gts_data = np.empty((num_samples,) + ray_trafo.domain.shape, dtype=np.float32)
        obs_data = np.empty((num_samples,) + ray_trafo.range.shape, dtype=np.float32)

    fbp_data = np.empty((num_samples,) + ray_trafo.domain.shape, dtype=np.float32)

    tmp_fbp = fbp.range.element()
    for i, (obs, gt) in zip(tqdm(range(num_samples), desc='generating cache'), dataset.generator(part)):
        fbp(obs, out=tmp_fbp)

        fbp_data[i, ...] = tmp_fbp

        if not only_fbp:
            obs_data[i, ...] = obs
            gts_data[i, ...] = gt

    if only_fbp:
        np.save(file_names[0], fbp_data)
    else:
        for filename, data in zip(file_names, [gts_data, obs_data, fbp_data]):
            np.save(filename, data)
