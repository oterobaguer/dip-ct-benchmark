from warnings import warn

import torch
import numpy as np

from torch.optim import Adam
from torch.nn import MSELoss

from odl.contrib.torch import OperatorModule
from dival import IterativeReconstructor

from dliplib.utils.losses import poisson_loss, tv_loss
from dliplib.utils.models import get_skip_model

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

MIN = -1000
MAX = 1000


class DeepImagePriorReconstructor(IterativeReconstructor):
    HYPER_PARAMS = {
        'lr':
            {'default': 1e-3,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'scales':
            {'default': 4,
             'choices': [3, 4, 5, 6, 7]},
        'channels':
            {'default': [128] * 5},
        'skip_channels':
            {'default': [4] * 5},
        'iterations':
            {'default': 5000,
             'range': [1, 50000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']}
    }
    """
    Deep Image Prior reconstructor similar to the one introduced in (cf. [1])
    References
    ----------
    .. [1] V. Lempitsky, A. Vedaldi, and D. Ulyanov, 2018,
           "Deep Image Prior".
           IEEE/CVF Conference on Computer Vision and Pattern Recognition.
           `doi:10.1109/CVPR.2018.00984
           <https://doi.org/10.1109/CVPR.2018.00984>`_
    """
    def __init__(self, ray_trafo, hyper_params=None, callback=None, callback_func=None, callback_func_interval=100, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        """

        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, callback=callback, **kwargs)

        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.domain_shape = ray_trafo.domain.shape
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval

    def get_activation(self, layer_index):
        return self.model.layer_output(self.net_input, layer_index)

    def _reconstruct(self, observation, *args, **kwargs):
        torch.random.manual_seed(10)
        lr = self.hyper_params['lr']
        gamma = self.hyper_params['gamma']
        scales = self.hyper_params['scales']
        channels = self.hyper_params['channels']
        iterations = self.hyper_params['iterations']
        skip_channels = self.hyper_params['skip_channels']
        loss_function = self.hyper_params['loss_function']


        output_depth = 1
        input_depth = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.net_input = 0.1 * torch.randn(input_depth, *self.reco_space.shape)[None].to(device)
        self.model = get_skip_model(input_depth,
                                    output_depth,
                                    channels=channels[:scales],
                                    skip_channels=skip_channels[:scales]).to(device)

        self.optimizer = Adam(self.model.parameters(), lr=lr)

        y_delta = torch.tensor(observation.asarray(), dtype=torch.float32)
        y_delta = y_delta.view(1, 1, *y_delta.shape)
        y_delta = y_delta.to(device)

        if loss_function == 'mse':
            criterion = MSELoss()
        elif loss_function == 'poisson':
            criterion = poisson_loss
        else:
            warn('Unknown loss function, falling back to MSE')
            criterion = MSELoss()

        best_loss = np.infty
        best_output = self.model(self.net_input).detach()

        for i in range(iterations):
            self.optimizer.zero_grad()
            output = self.model(self.net_input)
            loss = criterion(self.ray_trafo_module(output), y_delta) + gamma * tv_loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            self.optimizer.step()
            for p in self.model.parameters():
                p.data.clamp_(MIN, MAX)

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = output.detach()

            if (i % self.callback_func_interval == 0 or i == iterations-1) and self.callback_func is not None:
                self.callback_func(iteration=i, reconstruction=best_output[0, 0, ...].cpu().numpy(), loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(best_output[0, 0, ...].cpu().numpy()))

        return self.reco_space.element(best_output[0, 0, ...].cpu().numpy())

