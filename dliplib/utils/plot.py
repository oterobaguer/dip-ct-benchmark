import numpy as np
import matplotlib.pyplot as plt

from distutils.spawn import find_executable

from dival.measure import PSNR, SSIM
from dival.util.plot import plot_images


if find_executable('latex'):
    plt.rc('font', family='serif', serif='Computer Modern')
    plt.rc('text', usetex=True)
    plt.rc('xtick', labelsize=12)
    plt.rc('ytick', labelsize=12)
    plt.rc('axes', labelsize=12)


def plot_reconstructions(reconstructions, titles,  ray_trafo, obs, gt, save_name=None,
                         fig_size=(18, 4.5), cmap='pink'):
    """
    Plots a ground-truth and several reconstructions
    :param reconstructors: List of Reconstructor objects to compute the reconstructions
    :param test_data: Data to apply the reconstruction methods
    """
    psnrs = [PSNR(reco, gt) for reco in reconstructions]
    ssims = [SSIM(reco, gt) for reco in reconstructions]

    l2_error0 = np.sqrt(
        np.sum(np.power(ray_trafo(gt).asarray() - obs.asarray(), 2)))
    l2_error = [np.sqrt(np.sum(np.power(
        ray_trafo(reco).asarray() - obs.asarray(), 2))) for reco in reconstructions]

    # plot results
    im, ax = plot_images([gt, ] + reconstructions, fig_size=fig_size, rect=(0.0, 0.0, 1.0, 1.0),
                         xticks=[], yticks=[], vrange=(0.0, 0.9 * np.max(gt.asarray())), cbar=False,
                         interpolation='none', cmap=cmap)

    # set labels
    ax[0].set_title('Ground Truth')
    for j in range(len(reconstructions)):
        ax[j + 1].set_title(titles[j])
        ax[j + 1].set_xlabel('$\ell_2$ data error: {:.4f}\nPSNR: {:.1f}, SSIM: {:.2f}'
                             .format(l2_error[j], psnrs[j], ssims[j]))

    ax[0].set_xlabel('$\ell_2$ data error: {:.2f}'.format(l2_error0))

    plt.tight_layout()
    plt.tight_layout()

    if save_name:
        plt.savefig('%s.pdf' % save_name)
    plt.show()


def plot_reconstructors_tests(reconstructors, ray_trafo, test_data, save_name=None,
                              fig_size=(18, 4.5), cmap='pink'):
    """
    Plots a ground-truth and several reconstructions
    :param reconstructors: List of Reconstructor objects to compute the reconstructions
    :param test_data: Data to apply the reconstruction methods
    """
    titles = []
    for reconstructor in reconstructors:
        titles.append(reconstructor.name)

    for i in range(len(test_data)):
        y_delta, x = test_data[i]

        # compute reconstructions and psnr and ssim measures
        recos = [r.reconstruct(y_delta) for r in reconstructors]
        l2_error = [np.sqrt(np.sum(
            np.power(ray_trafo(reco).asarray() - y_delta.asarray(), 2))) for reco in recos]
        l2_error0 = np.sqrt(
            np.sum(np.power(ray_trafo(x).asarray() - y_delta.asarray(), 2)))

        psnrs = [PSNR(reco, x) for reco in recos]
        ssims = [SSIM(reco, x) for reco in recos]

        # plot results
        im, ax = plot_images([x, ] + recos, fig_size=fig_size, rect=(0.0, 0.0, 1.0, 1.0), ncols=4, nrows=-1,
                             xticks=[], yticks=[], vrange=(0.0, 0.9 * np.max(x.asarray())), cbar=False,
                             interpolation='none', cmap=cmap)

        # set labels
        ax = ax.reshape(-1)
        ax[0].set_title('Ground Truth')
        ax[0].set_xlabel('$\ell_2$ data error: {:.2f}'.format(l2_error0))
        for j in range(len(recos)):
            ax[j + 1].set_title(titles[j])
            ax[j + 1].set_xlabel('$\ell_2$ data error: {:.4f}\nPSNR: {:.1f}, SSIM: {:.2f}'
                                 .format(l2_error[j], psnrs[j], ssims[j]))

        plt.tight_layout()
        plt.tight_layout()

        if save_name:
            plt.savefig('%s-%d.pdf' % (save_name, i))
        plt.show()


def plot_iterations(recos, iters, save_name=None, fig_size=(18, 4.5), cmap='pink'):
    """
    Plot several iterates of an iterative method
    :param recos: List of reconstructions
    :param iters: Iteration numbers
    """
    im, ax = plot_images(recos, fig_size=fig_size, rect=(0.0, 0.0, 1.0, 1.0),
                         xticks=[], yticks=[], vrange=(0., 0.9), cbar=False,
                         interpolation='none', cmap=cmap)

    for i in range(len(iters)):
        ax[i].set_title('Iteration: %d' % iters[i])

    plt.tight_layout()
    plt.tight_layout()

    if save_name:
        plt.savefig('%s.pdf' % save_name)

    plt.show()
