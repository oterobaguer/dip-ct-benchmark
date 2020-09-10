import os
import torch
import tqdm

import numpy as np
import matplotlib.pyplot as plt

from distutils.spawn import find_executable
from warnings import warn
from dival import get_standard_dataset
from dival.datasets.dataset import Dataset
from dival.config import CONFIG as DIVAL_CONFIG
from odl.discr import nonuniform_partition
from odl.tomo.geometry import Parallel2dGeometry
from odl.tomo.operators import RayTransform


try:
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False


def select_hyper_best_parameters(results, measure='psnr'):
    """
    Selects the best hyper-parameter choice from a Dival ResultTable
    :param results: Dival ResultTable with one task containing several subtasks
    :param measure: Measure to use to select the best hyper-parameter choice
    :return: Dival hyper-parameters dictionary and the corresponding measure
    """
    best_choice = None
    best_measure = -np.inf

    assert len(results.results.loc[0]) == len(results.results)

    for _, row in results.results.iterrows():
        mean_measure = np.mean(row['measure_values'][measure])

        if mean_measure > best_measure:
            best_measure = mean_measure
            best_choice = row['misc'].get('hp_choice', dict())

    return best_choice, best_measure


def load_standard_dataset(dataset, impl=None, ordered=False):
    """
    Loads a Dival standard dataset.
    :param dataset: Name of the standard dataset
    :param impl: Backend for the Ray Transform
    :param ordered: Whether to order by patient id for 'lodopab' dataset
    :param angle_indices: Indices of the angles to include (default is all).
    :return: Dival dataset.
    """
    if impl is None:
        impl = 'astra_cpu'
        if torch.cuda.is_available():
            impl = 'astra_cuda'
    kwargs = {'impl': impl}
    if dataset == 'ellipses':
        kwargs['fixed_seeds'] = True
    # we do not use 'sorted_by_patient' here in order to be transparent to
    # `CachedDataset`, where a `ReorderedDataset` is handled specially
    # if dataset == 'lodopab':
        # kwargs['sorted_by_patient'] = ordered

    dataset_name = dataset.split('_')[0]
    dataset_out = get_standard_dataset(dataset_name, **kwargs)

    if dataset == 'lodopab_200':
        angles = list(range(0, 1000, 5))
        dataset_out = AngleSubsetDataset(dataset_out, angles)

    if dataset_name == 'lodopab' and ordered:
        idx = get_lodopab_idx_sorted_by_patient()
        dataset_ordered = ReorderedDataset(dataset_out, idx)
        dataset_ordered.ray_trafo = dataset_out.ray_trafo
        dataset_ordered.get_ray_trafo = dataset_out.get_ray_trafo
        dataset_out = dataset_ordered

    return dataset_out


def extract_tensorboard_scalars(log_dir=None, save_as_npz=None):
    if not TF_AVAILABLE:
        raise RuntimeError('Tensorflow could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir_path = os.path.join(BASE_DIR, 'utils/logs', log_dir)
    log_files = [f for f in os.listdir(log_dir_path)
                 if os.path.isfile(os.path.join(log_dir_path, f))]
    if len(log_files) == 0:
        raise FileNotFoundError('no file in log dir "{}"'.format(log_dir_path))
    elif len(log_files) > 1:
        warn('multiple files in log_dir "{}", choosing the one modified last'
             .format(log_dir))
        log_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(log_dir_path, f)),
            reverse=True)
    log_file = os.path.join(log_dir_path, log_files[0])

    def my_summary_iterator(path):
        for r in tf_record.tf_record_iterator(path):
            yield event_pb2.Event.FromString(r)

    values = {}
    for event in tqdm.tqdm(my_summary_iterator(log_file)):
        if event.WhichOneof('what') != 'summary':
            continue
        step = event.step
        for value in event.summary.value:
            if value.HasField('simple_value'):
                tag = value.tag.replace('/', '_').lower()
                values.setdefault(tag, []).append((step, value.simple_value))
    scalars = {}
    for k in values.keys():
        v = np.asarray(values[k])
        steps, steps_counts = np.unique(v[:, 0], return_counts=True)
        n_per_step = steps_counts[0]
        assert np.all(steps_counts == n_per_step)
        scalars[k + '_steps'] = steps
        scalars[k + '_scalars'] = v[n_per_step-1::n_per_step, 1]  # last of
        #                                                           each step

    if save_as_npz is not None:
        np.savez(save_as_npz, **scalars)

    return scalars


def set_use_latex():
    if find_executable('latex'):
        plt.rc('font', family='serif', serif='Computer Modern')
        plt.rc('text', usetex=True)
        plt.rc('xtick', labelsize=12)
        plt.rc('ytick', labelsize=12)
        plt.rc('axes', labelsize=12)
    else:
        warn('Latex not available on this device')


class ReorderedDataset(Dataset):
    def __init__(self, dataset, idx):
        """
        Parameters
        ----------
        dataset : `Dataset`
            Dataset to take the samples from. Must support random access.
        idx : dict of array-like
            Indices into the original dataset for each part.
            Each array-like must have (at least) the same length as the part.
        """
        assert dataset.supports_random_access()
        self.dataset = dataset
        self.idx = idx
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.random_access = True
        self.num_elements_per_sample = (
            self.dataset.get_num_elements_per_sample())
        super().__init__(space=self.dataset.space)
        self.shape = self.dataset.get_shape()

    def get_sample(self, index, part='train', out=None):
        sample = self.dataset.get_sample(
            self.idx[part][index], part=part, out=out)
        return sample


def get_lodopab_idx_sorted_by_patient():
    idx = {}
    for part in ('train', 'validation', 'test'):
        data_path = DIVAL_CONFIG['lodopab_dataset']['data_path']
        ids = np.loadtxt(
            os.path.join(data_path, 'patient_ids_rand_{}.csv'.format(part)),
            dtype=np.int)
        idx[part] = np.argsort(ids, kind='stable')
    return idx


class AngleSubsetDataset(Dataset):
    def __init__(self, dataset, angle_indices, impl=None):
        """
        Parameters
        ----------
        dataset : `Dataset`
            Basis CT dataset.
            Requirements:
                - sample elements are ``(observation, ground_truth)``
                - :meth:`get_ray_trafo` gives corresponding ray transform.
        angle_indices : array-like or slice
            Indices of the angles to use from the observations.
        impl : {``'skimage'``, ``'astra_cpu'``, ``'astra_cuda'``},\
                optional
            Implementation passed to :class:`odl.tomo.RayTransform` to
            construct :attr:`ray_trafo`.
        """
        self.dataset = dataset
        self.angle_indices = (angle_indices if isinstance(angle_indices, slice)
                              else np.asarray(angle_indices))
        self.train_len = self.dataset.get_len('train')
        self.validation_len = self.dataset.get_len('validation')
        self.test_len = self.dataset.get_len('test')
        self.random_access = self.dataset.supports_random_access()
        self.num_elements_per_sample = (
            self.dataset.get_num_elements_per_sample())
        orig_geometry = self.dataset.get_ray_trafo(impl=impl).geometry
        apart = nonuniform_partition(
            orig_geometry.angles[self.angle_indices])
        self.geometry = Parallel2dGeometry(
            apart=apart, dpart=orig_geometry.det_partition)
        orig_shape = self.dataset.get_shape()
        self.shape = ((apart.shape[0], orig_shape[0][1]), orig_shape[1])
        self.space = (None, self.dataset.space[1])  # preliminary, needed for
        # call to get_ray_trafo
        self.ray_trafo = self.get_ray_trafo(impl=impl)
        super().__init__(space=(self.ray_trafo.range, self.dataset.space[1]))

    def get_ray_trafo(self, **kwargs):
        """
        Return the ray transform that matches the subset of angles specified to
        the constructor via `angle_indices`.
        """
        return RayTransform(self.space[1], self.geometry, **kwargs)

    def generator(self, part='train'):
        for (obs, gt) in self.dataset.generator(part=part):
            yield (self.space[0].element(obs[self.angle_indices]), gt)

    def get_sample(self, index, part='train', out=None):
        if out is None:
            out = (True, True)
        (out_obs, out_gt) = out
        out_basis = (out_obs is not False, out_gt)
        obs_basis, gt = self.dataset.get_sample(index, part=part,
                                                out=out_basis)
        if isinstance(out_obs, bool):
            obs = (self.space[0].element(obs_basis[self.angle_indices])
                   if out_obs else None)
        else:
            out_obs[:] = obs_basis[self.angle_indices]
            obs = out_obs
        return (obs, gt)

    def get_samples(self, key, part='train', out=None):
        if out is None:
            out = (True, True)
        (out_obs, out_gt) = out
        out_basis = (out_obs is not False, out_gt)
        obs_arr_basis, gt_arr = self.dataset.get_samples(key, part=part,
                                                         out=out_basis)
        if isinstance(out_obs, bool):
            obs_arr = obs_arr_basis[:, self.angle_indices] if out_obs else None
        else:
            out_obs[:] = obs_arr_basis[:, self.angle_indices]
            obs_arr = out_obs
        return (obs_arr, gt_arr)
