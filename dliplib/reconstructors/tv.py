from dliplib.utils.models import get_skip_model
from dliplib.utils.losses import poisson_loss, tv_loss
from dival import IterativeReconstructor
from odl.contrib.torch import OperatorModule
from torch.nn import MSELoss
from torch.optim import Adam
import numpy as np
import torch
import odl

from warnings import warn
from odl.tomo import fbp_op
from dival import Reconstructor


class TVReconstructor(Reconstructor):
    HYPER_PARAMS = {
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1.0],
             'grid_search_options': {'num_samples': 20}},
        'iterations':
            {'default': 200,
             'range': [1, 200]}
    }

    def __init__(self, ray_trafo, hyper_params=None, iterations=None, gamma=None, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        """

        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, **kwargs)

        self.ray_trafo = ray_trafo
        self.domain_shape = ray_trafo.domain.shape
        self.opnorm = odl.power_method_opnorm(ray_trafo)
        self.fbp_op = fbp_op(
            ray_trafo, frequency_scaling=0.1, filter_type='Hann')

        if iterations is not None:
            self.iterations = iterations
            if kwargs.get('hyper_params', {}).get('iterations') is not None:
                warn("hyper parameter 'iterations' overridden by constructor argument")

        if gamma is not None:
            self.gamma = gamma
            if kwargs.get('hyper_params', {}).get('gamma') is not None:
                warn("hyper parameter 'gamma' overridden by constructor argument")

    def get_iterations(self):
        return self.hyper_params['iterations']

    def set_iterations(self, iterations):
        self.hyper_params['iterations'] = iterations

    iterations = property(get_iterations, set_iterations)

    def get_gamma(self):
        return self.hyper_params['gamma']

    def set_gamma(self, gamma):
        self.hyper_params['gamma'] = gamma

    gamma = property(get_gamma, set_gamma)

    def _reconstruct(self, observation, *args, **kwargs):
        xspace = self.ray_trafo.domain
        yspace = self.ray_trafo.range

        grad = odl.Gradient(self.ray_trafo.domain)

        # Assemble all operators into a list.
        lin_ops = [self.ray_trafo, grad]

        # Create functionals for the l2 distance and l1 norm.
        g_funcs = [odl.solvers.L2NormSquared(yspace).translated(observation),
                   self.gamma * odl.solvers.L1Norm(grad.range)]

        # Functional of the bound constraint 0 <= x <= 1
        f = odl.solvers.IndicatorBox(xspace, 0, 1)

        # Find scaling constants so that the solver converges.
        # See the douglas_rachford_pd documentation for more information.
        xstart = self.fbp_op(observation)
        opnorm_A = odl.power_method_opnorm(self.ray_trafo, xstart=xstart)
        opnorm_grad = odl.power_method_opnorm(grad, xstart=xstart)
        sigma = [1 / opnorm_A ** 2, 1 / opnorm_grad ** 2]
        tau = 1.0

        # Solve using the Douglas-Rachford Primal-Dual method
        x = xspace.zero()
        odl.solvers.douglas_rachford_pd(x, f, g_funcs, lin_ops,
                                        tau=tau,
                                        sigma=sigma,
                                        niter=self.iterations)
        return x


class TVAdamReconstructor(IterativeReconstructor):
    HYPER_PARAMS = {
        'lr':
            {'default': 1e-3,
             'range': [1e-5, 1e-1]},
        'gamma':
            {'default': 1e-4,
             'range': [1e-7, 1e-0],
             'grid_search_options': {'num_samples': 20}},
        'iterations':
            {'default': 5000,
             'range': [1, 50000]},
        'loss_function':
            {'default': 'mse',
             'choices': ['mse', 'poisson']}
    }
    """
    Reconstructor minimizing a TV-functional with the Adam optimizer.
    """

    def __init__(self, ray_trafo, hyper_params=None, callback=None,
                 callback_func=None, callback_func_interval=100, **kwargs):
        """
        Parameters
        ----------
        ray_trafo : `odl.tomo.operators.RayTransform`
            The forward operator
        """

        super().__init__(
            reco_space=ray_trafo.domain, observation_space=ray_trafo.range,
            hyper_params=hyper_params, callback=callback, **kwargs)

        self.fbp_op = fbp_op(
            ray_trafo, frequency_scaling=0.1, filter_type='Hann')
        self.callback_func = callback_func
        self.ray_trafo = ray_trafo
        self.ray_trafo_module = OperatorModule(self.ray_trafo)
        self.domain_shape = ray_trafo.domain.shape
        self.callback_func = callback_func
        self.callback_func_interval = callback_func_interval

    def _reconstruct(self, observation, *args, **kwargs):
        torch.random.manual_seed(10)
        lr = self.hyper_params['lr']
        gamma = self.hyper_params['gamma']
        iterations = self.hyper_params['iterations']
        loss_function = self.hyper_params['loss_function']

        self.net_input = torch.tensor(self.fbp_op(observation))[None]
        self.net_input.requires_grad = True

        self.model = torch.nn.Identity()
        self.optimizer = Adam([self.net_input], lr=lr)

        y_delta = torch.tensor(observation.asarray(), dtype=torch.float32)
        y_delta = y_delta.view(1, *y_delta.shape)
        y_delta = y_delta

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
            loss = criterion(self.ray_trafo_module(output),
                             y_delta) + gamma * tv_loss(output)
            loss.backward()

            self.optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_output = output.detach()

            if (i % self.callback_func_interval == 0 or i == iterations-1) and self.callback_func is not None:
                self.callback_func(
                    iteration=i, reconstruction=best_output[0, ...].cpu().numpy(), loss=best_loss)

            if self.callback is not None:
                self.callback(self.reco_space.element(
                    best_output[0, ...].cpu().numpy()))

        return self.reco_space.element(best_output[0, ...].cpu().numpy())
