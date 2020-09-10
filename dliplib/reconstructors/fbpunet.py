import torch

import numpy as np
import torch.nn as nn
from warnings import warn
from copy import deepcopy

from odl.tomo import fbp_op
from dliplib.reconstructors.base import BaseLearnedReconstructor
from dliplib.utils.models import get_unet_model


class FBPUNetReconstructor(BaseLearnedReconstructor):
    HYPER_PARAMS = deepcopy(BaseLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'channels': {
            'default': (32, 32, 64, 64, 128, 128),
            'retrain': True
        },
        'filter_type': {
            'default': 'Hann',
            'retrain': True
        },
        'frequency_scaling': {
            'default': 1.0,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
        'init_bias_zero': {
            'default': True,
            'retrain': True
        },
        'lr': {
            'default': 0.001,
            'retrain': True
        },
        'scheduler': {
            'default': 'cosine',
            'choices': ['base', 'cosine'],  # 'base': inherit
            'retrain': True
        },
        'lr_min': {  # only used if 'cosine' scheduler is selected
            'default': 1e-4,
            'retrain': True
        }
    })
    """
    CT Reconstructor applying filtered back-projection followed by a
    postprocessing U-Net (cf. [1]_).
    References
    ----------
    .. [1] K. H. Jin, M. T. McCann, E. Froustey, et al., 2017,
           "Deep Convolutional Neural Network for Inverse Problems in Imaging".
           IEEE Transactions on Image Processing.
           `doi:10.1109/TIP.2017.2713099
           <https://doi.org/10.1109/TIP.2017.2713099>`_
    """

    def __init__(self, ray_trafo, filter_type=None, frequency_scaling=None,
                 scales=None, epochs=None, batch_size=None, lr=None,
                 skip_channels=None, num_data_loader_workers=8, use_cuda=True,
                 show_pbar=True, fbp_impl='astra_cuda', hyper_params=None,
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        scales : int, optional
            Number of scales in the U-Net (a hyper parameter).
        epochs : int, optional
            Number of epochs to train (a hyper parameter).
        batch_size : int, optional
            Batch size (a hyper parameter).
        num_data_loader_workers : int, optional
            Number of parallel workers to use for loading data.
        use_cuda : bool, optional
            Whether to use cuda for the U-Net.
        show_pbar : bool, optional
            Whether to show tqdm progress bars during the epochs.
        fbp_impl : str, optional
            The backend implementation passed to
            :class:`odl.tomo.RayTransform` in case no `ray_trafo` is specified.
            Then ``dataset.get_ray_trafo(impl=fbp_impl)`` is used to get the
            ray transform and FBP operator.
        """

        super().__init__(ray_trafo, epochs=epochs, batch_size=batch_size, lr=lr,
                         num_data_loader_workers=num_data_loader_workers, use_cuda=use_cuda, show_pbar=show_pbar,
                         fbp_impl=fbp_impl, hyper_params=hyper_params, **kwargs)

        if scales is not None:
            self.scales = scales
            if kwargs.get('hyper_params', {}).get('scales') is not None:
                warn("hyper parameter 'scales' overridden by constructor argument")

        if skip_channels is not None:
            self.skip_channels = skip_channels
            if kwargs.get('hyper_params', {}).get('skip_channels') is not None:
                warn("hyper parameter 'skip_channels' overridden by constructor argument")

        if filter_type is not None:
            self.filter_type = filter_type
            if kwargs.get('hyper_params', {}).get('filter_type') is not None:
                warn("hyper parameter 'filter_type' overridden by constructor argument")

        if frequency_scaling is not None:
            self.frequency_scaling = frequency_scaling
            if kwargs.get('hyper_params', {}).get('frequency_scaling') is not None:
                warn(
                    "hyper parameter 'frequency_scaling' overridden by constructor argument")

        # TODO: update fbp_op when the hyper parameters change?
        self.fbp_op = fbp_op(ray_trafo, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)

    def init_model(self):
        self.fbp_op = fbp_op(self.ray_trafo, filter_type=self.filter_type,
                             frequency_scaling=self.frequency_scaling)
        self.model = get_unet_model(scales=self.scales,
                                    skip=self.skip_channels,
                                    channels=self.channels,
                                    use_sigmoid=self.use_sigmoid)

        if self.init_bias_zero:
            def weights_init(m):
                if isinstance(m, torch.nn.Conv2d):
                    m.bias.data.fill_(0.0)
            self.model.apply(weights_init)

        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    def init_scheduler(self, dataset_train):
        if self.scheduler.lower() == 'cosine':
            self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.epochs,
                eta_min=self.lr_min)
        else:
            super().init_scheduler(dataset_train)

    def _reconstruct(self, observation):
        self.model.eval()
        fbp = self.fbp_op(observation)
        fbp_tensor = torch.from_numpy(
            np.asarray(fbp)[None, None]).to(self.device)
        reco_tensor = self.model(fbp_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)
