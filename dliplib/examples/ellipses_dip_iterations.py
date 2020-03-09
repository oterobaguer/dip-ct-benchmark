import matplotlib.pyplot as plt

from dival.measure import PSNR

from dliplib.utils import Params
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_iterations
from dliplib.reconstructors.dip import DeepImagePriorReconstructor


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 5)
obs, gt = test_data[0]


def plot_result(result, gt, iteration):
    psnr = PSNR(result, gt)
    plt.imshow(result, cmap='bone')
    plt.title('%d: %.3f' % (iteration, psnr))
    plt.axis('off')
    plt.show()


recos = []
iters = []


def callback_func(iteration, reconstruction, loss):
    recos.append(reconstruction)
    iters.append(iteration)
    print(iteration)
    # plot_result(reconstruction, gt, iteration)


# load hyper-parameters and create reconstructor
params = Params.load('ellipses_diptv')
params.iterations = 2001
reconstructor = DeepImagePriorReconstructor(ray_trafo=dataset.ray_trafo, hyper_params=params.dict,
                                            name='DIP+TV', callback_func_interval=10)
reconstructor.callback_func = callback_func
result = reconstructor.reconstruct(obs)

plot_iterations([recos[0], recos[5], recos[15], recos[45], recos[-1]],
                [iters[0], iters[5], iters[15], iters[45], iters[-1]],
                save_name='ellipses_dip_iterations',
                fig_size=(10, 2.5))