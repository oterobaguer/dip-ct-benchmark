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
    if impl is None:
        impl = 'astra_cpu'
        if torch.cuda.is_available():
            impl = 'astra_cuda'
    if dataset == 'ellipses':
        return get_standard_dataset('ellipses', fixed_seeds=True, impl=impl)
    else:
        orig_dataset = get_standard_dataset(dataset, impl=impl)
        if ordered:
            idx = get_lodopab_idx_sorted_by_patient()
            dataset = ReorderedDataset(orig_dataset, idx)
            dataset.ray_trafo = orig_dataset.ray_trafo
            return dataset
        else:
            return orig_dataset


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
