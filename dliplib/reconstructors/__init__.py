import os
import numpy as np

from dival.reconstructors.odl_reconstructors import FBPReconstructor

from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.reconstructors.dip_initial import DeepImagePriorInitialReconstructor
from dliplib.reconstructors.fbpunet import FBPUNetReconstructor
from dliplib.reconstructors.iradonmap import IRadonMapReconstructor
from dliplib.reconstructors.learnedgd import LearnedGDReconstructor
from dliplib.reconstructors.learnedpd import LearnedPDReconstructor
from dliplib.reconstructors.tv import TVReconstructor
from dliplib.utils import Params
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.weights import load_weights


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def fbpunet_reconstructor(dataset='ellipses', size_part=1.0, pretrained=True, name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :param size_part: Can be one of: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    :return: The FBP+UNet method trained on the specified dataset and size
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_fbpunet_{}'.format(dataset, size_part))
        if name is None:
            name = 'FBP+UNet ({} $\%$)'.format(100 * size_part)


        reconstructor = FBPUNetReconstructor(standard_dataset.ray_trafo,
                                             hyper_params=params.dict,
                                             name=name)
        if pretrained:
            load_weights(reconstructor, '{}_fbpunet_{}'.format(dataset, size_part))
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor has not been trained with the selected data_size')


def learnedpd_reconstructor(dataset='ellipses', size_part=1.0, pretrained=True, name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :param size_part: Can be one of: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    :return: The Learned Primal-Dual method trained on the specified dataset and size
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_learnedpd_{}'.format(dataset, size_part))
        if name is None:
            name = 'Learned PD ({} $\%$)'.format(100 * size_part)
        reconstructor = LearnedPDReconstructor(standard_dataset.ray_trafo,
                                               hyper_params=params.dict,
                                               name=name)
        if pretrained:
            load_weights(reconstructor, '{}_learnedpd_{}'.format(dataset, size_part))
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor has not been trained with the selected data_size')


def learnedgd_reconstructor(dataset='ellipses', size_part=1.0, pretrained=True, name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :param size_part: Can be one of: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    :return: The Learned Gradient Descent method trained on the specified dataset and size
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_learnedgd_{}'.format(dataset, size_part))
        if name is None:
            name = 'Learned GD ({} $\%$)'.format(100 * size_part)
        reconstructor = LearnedGDReconstructor(standard_dataset.ray_trafo,
                                               hyper_params=params.dict,
                                               name=name)
        if pretrained:
            load_weights(reconstructor, '{}_learnedgd_{}'.format(dataset, size_part))
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor has not been trained with the selected data_size')


def iradonmap_reconstructor(dataset='ellipses', size_part=1.0, pretrained=True, name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :param size_part: Can be one of: [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    :return: The iRadonMap method trained on the specified dataset and size
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_iradonmap_{}'.format(dataset, size_part))
        if name is None:
            name = 'iRadonMap ({} $\%$)'.format(100 * size_part)
        coord_mat = None
        try:
            coord_mat = np.load(os.path.join(BASE_DIR, 'reconstructors',
                                '{}_iradonmap_coord_mat.npy'.format(dataset)))
        except FileNotFoundError:
            pass
        reconstructor = IRadonMapReconstructor(standard_dataset.ray_trafo,
                                               hyper_params=params.dict,
                                               name=name, coord_mat=coord_mat)
        if pretrained:
            load_weights(reconstructor, '{}_iradonmap_{}'.format(dataset, size_part))
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor has not been trained with the selected data_size')


def dip_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: The Deep Image Prior (DIP) method for the specified dataset
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_dip'.format(dataset))
        if name is None:
            name = 'DIP'
        reconstructor = DeepImagePriorReconstructor(standard_dataset.ray_trafo,
                                                    hyper_params=params.dict,
                                                    name=name)
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')


def diptv_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: The Deep Image Prior (DIP) + TV method for the specified dataset
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_diptv'.format(dataset))
        if name is None:
            name = 'DIP + TV'
        reconstructor = DeepImagePriorReconstructor(standard_dataset.ray_trafo,
                                                    hyper_params=params.dict,
                                                    name=name)
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')


def learnedpd_dip_reconstructor(dataset='ellipses', size_part=1.0, name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: The Deep Image Prior (DIP) method for the specified dataset
    """
    try:
        standard_dataset = load_standard_dataset(dataset)
        params = Params.load('{}_learnedpd_{}'.format(dataset, size_part))
        learned_pd = LearnedPDReconstructor(standard_dataset.ray_trafo,
                                            hyper_params=params.dict,
                                            name='Learned PD ({}%)'.format(size_part * 100))
        load_weights(learned_pd, '{}_learnedpd_{}'.format(dataset, size_part))

        # load hyper-parameters and create reconstructor
        if name is None:
            name = 'Learned PD ({} $\%$) + DIP'.format(100 * size_part)
        params = Params.load('{}_learnedpd_dip_{}'.format(dataset, size_part))
        reconstructor = DeepImagePriorInitialReconstructor(standard_dataset.ray_trafo,
                                                           ini_reco=learned_pd,
                                                           hyper_params=params.dict,
                                                           name=name)
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')


def fbp_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: Filtered back projection reconstructor for the specified dataset
    """
    try:
        params = Params.load('{}_fbp'.format(dataset))
        standard_dataset = load_standard_dataset(dataset)
        if name is None:
            name = 'FBP'
        reconstructor = FBPReconstructor(standard_dataset.ray_trafo,
                                         hyper_params=params.dict,
                                         name=name)
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')


def tv_reconstructor(dataset='ellipses', name=None):
    """
    :param dataset: Can be 'ellipses' or 'lodopab'
    :return: TV reconstructor for the specified dataset
    """
    try:
        params = Params.load('{}_tv'.format(dataset))
        standard_dataset = load_standard_dataset(dataset)
        if name is None:
            name = 'TV'
        reconstructor = TVReconstructor(standard_dataset.ray_trafo,
                                        hyper_params=params.dict,
                                        name=name)
        return reconstructor
    except Exception as e:
        raise Exception('The reconstructor doesn\'t exist')