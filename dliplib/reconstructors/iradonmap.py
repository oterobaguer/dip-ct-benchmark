# -*- coding: utf-8 -*-
import torch

import numpy as np
import torch.nn as nn
from warnings import warn
from copy import deepcopy

from dliplib.reconstructors.base import BaseLearnedReconstructor
from dliplib.utils.models import get_iradonmap_model


class IRadonMapReconstructor(BaseLearnedReconstructor):
    HYPER_PARAMS = deepcopy(BaseLearnedReconstructor.HYPER_PARAMS)
    HYPER_PARAMS.update({
        'scales': {
            'default': 5,
            'retrain': True
        },
        'epochs': {
            'default': 20,
            'retrain': True
        },
        'lr': {
            'default': 0.01,
            'retrain': True
        },
        'skip_channels': {
            'default': 4,
            'retrain': True
        },
        'batch_size': {
            'default': 64,
            'retrain': True
        },
        'fully_learned': {
            'default': False,
            'retrain': True
        },
        'use_sigmoid': {
            'default': False,
            'retrain': True
        },
    })
    """
    CT Reconstructor that learns a fully connected layer for filtering along
    the axis of the detector pixels s, followed by the backprojection
    (segment 1). After that, a residual CNN acts as a post-processing net
    (segment 2). We use the U-Net from the FBPUnet model.

    In the original paper [1], a learned version of the back-
    projection layer (sinusoidal layer) is used. This layer introduces a lot
    more parameters. Therefore, we added an option to directly use the operator
    in our implementation. Additionally, we drop the tanh activation after
    the first fully connectect layer, due to bad performance.

    In any configuration, the iRadonMap has less parameters than an
    Automap network [2].

    References
    ----------
    .. [1] J. He and J. Ma, 2018,
           "Radon Inversion via Deep Learning".
           arXiv preprint.
           `arXiv:1808.03015v1
           <https://arxiv.org/abs/1808.03015>`_
    .. [2] B. Zhu, J. Z. Liu, S. F. Cauly et al., 2018,
           "Image Reconstruction by Domain-Transform Manifold Learning".
           Nature 555, 487--492.
           `doi:10.1038/nature25988
           <https://doi.org/10.1038/nature25988>`_
    """

    def __init__(self, ray_trafo, fully_learned=None, scales=None,
                 epochs=None, batch_size=None, lr=None, skip_channels=None,
                 num_data_loader_workers=8, use_cuda=True, show_pbar=True,
                 fbp_impl='astra_cuda', hyper_params=None, coord_mat=None,
                 **kwargs):
        """
        Parameters
        ----------
        ray_trafo : :class:`odl.tomo.RayTransform`
            Ray transform from which the FBP operator is constructed.
        fully_learned : bool, optional
            Learn the backprojection operator or take the fixed one from astra.
        epochs : int, optional
            Number of epochs to train (a hyper parameter).
        batch_size : int, optional
            Batch size (a hyper parameter).
        num_data_loader_workers : int, optional
            Number of parallel workers to use for loading data.
        use_cuda : bool, optional
            Whether to use cuda for the model.
        show_pbar : bool, optional
            Whether to show tqdm progress bars during the epochs.
        fbp_impl : str, optional
            The backend implementation passed to
            :class:`odl.tomo.RayTransform` in case no `ray_trafo` is specified.
            Then ``dataset.get_ray_trafo(impl=fbp_impl)`` is used to get the
            ray transform and FBP operator.
        coord_mat : array, optional
            Precomputed coordinate matrix for the `LearnedBackprojection`.
            This option is provided for performance optimization.
            If `None` is passed, the matrix is computed in :meth:`init_model`.
        """

        super().__init__(ray_trafo, epochs=epochs, batch_size=batch_size,
                         lr=lr,
                         num_data_loader_workers=num_data_loader_workers,
                         use_cuda=use_cuda, show_pbar=show_pbar,
                         fbp_impl=fbp_impl, hyper_params=hyper_params,
                         **kwargs)

        if scales is not None:
            hyper_params['scales'] = scales
            if kwargs.get('hyper_params', {}).get('scales') is not None:
                warn("hyper parameter 'scales' overridden by constructor " +
                     "argument")

        if skip_channels is not None:
            hyper_params['skip_channels'] = skip_channels
            if kwargs.get('hyper_params', {}).get('skip_channels') is not None:
                warn("hyper parameter 'skip_channels' overridden by " +
                     "constructor argument")

        if fully_learned is not None:
            hyper_params['fully_learned'] = fully_learned
            if kwargs.get('hyper_params', {}).get('fully_learned') is not None:
                warn("hyper parameter 'fully_learned' overridden by " +
                     "constructor argument")

        self.coord_mat = coord_mat

    def get_skip_channels(self):
        return self.hyper_params['skip_channels']

    def set_skip_channels(self, skip_channels):
        self.hyper_params['skip_channels'] = skip_channels

    skip_channels = property(get_skip_channels, set_skip_channels)

    def get_scales(self):
        return self.hyper_params['scales']

    def set_scales(self, scales):
        self.hyper_params['scales'] = scales

    scales = property(get_scales, set_scales)

    def get_fully_learned(self):
        return self.hyper_params['fully_learned']

    def set_fully_learned(self, fully_learned):
        self.hyper_params['fully_learned'] = fully_learned

    fully_learned = property(get_fully_learned, set_fully_learned)

    def init_model(self):
        self.model = get_iradonmap_model(
                ray_trafo=self.ray_trafo, fully_learned=self.fully_learned,
                scales=self.scales, skip=self.skip_channels,
                use_sigmoid=self.hyper_params['use_sigmoid'],
                coord_mat=self.coord_mat)
        if self.use_cuda:
            self.model = nn.DataParallel(self.model).to(self.device)

    # def init_optimizer(self, dataset_train):
    #     self.optimizer = torch.optim.RMSprop(self.model.parameters(),
    #                                          lr=self.lr, momentum=0.9)

    def _reconstruct(self, observation):
        self.model.eval()
        obs_tensor = torch.from_numpy(
                np.asarray(observation)[None, None]).to(self.device)
        reco_tensor = self.model(obs_tensor)
        reconstruction = reco_tensor.cpu().detach().numpy()[0, 0]
        return self.reco_space.element(reconstruction)
