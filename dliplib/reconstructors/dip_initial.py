import copy
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


class DeepImagePriorInitialReconstructor(IterativeReconstructor):
    HYPER_PARAMS = {
        'lr1':
            {'default': 1e-2,
             'range': [1e-5, 1e-1]},
        'lr2':
            {'default': 1e-2,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-2],
             'grid_search_options': {'num_samples': 20}},
        'scales':
            {'default': 4,
             'choices': [3, 4, 5, 6, 7]},
        'channels':
            {'default': [128] * 5},
        'skip_channels':
            {'default': [4] * 5},
        'initial_iterations':
            {'default': 2000,
             'range': [1, 10000]},
        'iterations':
            {'default': 2000,
             'range': [1, 10000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']}
    }

    def __init__(self, ray_trafo, ini_reco, hyper_params=None, callback=None, callback_func=None,
                 callback_func_interval=100, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        ini_reco: `dival.Reconstructor`
            Reconstructor used for the initial reconstruction
        """
        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, callback=callback, **kwargs)

        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)

        self.domain_shape = ray_trafo.domain.shape
        self.ini_reco = ini_reco
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval

    def _reconstruct(self, observation, *args, **kwargs):
        torch.random.manual_seed(1)
        lr1 = self.hyper_params['lr1']
        lr2 = self.hyper_params['lr2']
        gamma = self.hyper_params['gamma']
        scales = self.hyper_params['scales']
        channels = self.hyper_params['channels']
        initial_iterations = self.hyper_params['initial_iterations']
        iterations = self.hyper_params['iterations']
        skip_channels = self.hyper_params['skip_channels']
        loss_function = self.hyper_params['loss_function']

        output_depth = 1
        input_depth = 1
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.model = get_skip_model(input_depth,
                                    output_depth,
                                    channels=channels[:scales],
                                    skip_channels=skip_channels[:scales]).to(device)

        y_delta = torch.from_numpy(np.asarray(observation)[None, None])
        y_delta = y_delta.to(device)

        x0 = torch.tensor(self.ini_reco.reconstruct(observation).asarray(), dtype=torch.float32)
        x0 = x0.view(1, 1, *x0.shape)
        x0 = x0.to(device)

        noise = torch.randn(1, *self.reco_space.shape)[None].to(device)
        noise_saved = noise.clone()

        self.optimizer = Adam(self.model.parameters(), lr=lr1)
        mse = MSELoss()

        # Initial phase for training for \Theta
        best_loss = np.infty
        best_model_wts = copy.deepcopy(self.model.state_dict())

        for i in range(initial_iterations):
            self.net_input = noise_saved
            output = self.model(self.net_input)
            self.optimizer.zero_grad()
            loss = mse(output, x0)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1)
            loss.backward()
            self.optimizer.step()

            # current reconstruction
            x_rec = output.detach().cpu().numpy()

            # call custom callback
            # TODO: remove this and use proper ODL callbacks
            if (i % self.callback_func_interval == 0 or i == initial_iterations-1) and self.callback_func:
                self.callback_func(iteration=i, reconstruction=x_rec[0, 0, ...], loss=loss.item())

            # if the loss improved update the current best reconstruction
            # and save the model parameters
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_model_wts = copy.deepcopy(self.model.state_dict())

        # load the stored best parameters
        self.model.load_state_dict(best_model_wts)
        # restart optimizer
        self.optimizer = Adam(self.model.parameters(), lr=lr2, amsgrad=True)

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
            net_input = noise_saved
            output = self.model(net_input)
            self.optimizer.zero_grad()
            loss = criterion(self.ray_trafo_module(output), y_delta) + gamma * tv_loss(output)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1)
            self.optimizer.step()

            # if the loss improved update the current best reconstruction
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = output.detach()

            if (i % self.callback_func_interval == 0 or i == iterations - 1) and self.callback_func is not None:
                self.callback_func(iteration=i, reconstruction=best_output[0, 0, ...].cpu().numpy(), loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(best_output[0, 0, ...].cpu().numpy()))

        return self.reco_space.element(best_output[0, 0, ...].cpu().numpy())
