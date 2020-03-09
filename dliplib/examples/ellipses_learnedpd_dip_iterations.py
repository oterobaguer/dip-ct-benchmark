import matplotlib.pyplot as plt

from dival.measure import PSNR, SSIM

from dliplib.reconstructors import learnedpd_dip_reconstructor
from dliplib.utils.helper import load_standard_dataset, set_use_latex


set_use_latex()

# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 5)
ray_trafo = dataset.ray_trafo


# obs, gt = test_data[314]
obs, gt = test_data[0]
# obs, gt = test_data[100]
# load hyper-parameters and create reconstructor


def plot_result(result, gt, iteration):
    psnr = PSNR(result, gt)
    ssim = SSIM(result, gt)
    plt.imshow(result, cmap='gray')
    plt.axis('off')
    plt.title('%d: PSNR: %.2f SSIM: %.4f' % (iteration, psnr, ssim))
    plt.show()


def callback(iteration, reconstruction, loss):
    print(loss)
    plot_result(reconstruction, gt, iteration)


data_size = 0.001
reconstructor = learnedpd_dip_reconstructor('ellipses', data_size)

reconstructor.callback_func = callback
result = reconstructor.reconstruct(obs)

plot_result(result, gt, -1)