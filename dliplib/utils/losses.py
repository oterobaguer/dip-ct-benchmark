import torch
import numpy as np


def tv_loss(x):
    """
    Isotropic TV loss similar to the one in (cf. [1])
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Total_variation_denoising
    """
    dh = torch.abs(x[..., :, 1:] - x[..., :, :-1])
    dw = torch.abs(x[..., 1:, :] - x[..., :-1, :])
    return torch.sum(dh[..., :-1, :] + dw[..., :, :-1])


def poisson_loss(y_pred, y_true, photons_per_pixel=4096, mu_max=3071*(20-0.02)/1000+20):
    """
    Loss corresponding to Poisson regression (cf. [1]). The default parameters
    are based on the LoDoPaB dataset creation (cf. [2])
    References
    ----------
    .. [1] https://en.wikipedia.org/wiki/Poisson_regression
    .. [2] https://github.com/jleuschn/lodopab_tech_ref/blob/master/create_dataset.py
    """

    def get_photons(y):
        y = torch.exp(-y * mu_max) * photons_per_pixel
        return y

    def get_photons_log(y):
        y = -y * mu_max + np.log(photons_per_pixel)
        return y

    y_true_photons = get_photons(y_true)
    y_pred_photons = get_photons(y_pred)
    y_pred_photons_log = get_photons_log(y_pred)

    return torch.sum(y_pred_photons - y_true_photons * y_pred_photons_log)
