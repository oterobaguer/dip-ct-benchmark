import matplotlib.pyplot as plt

from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.utils.helper import load_standard_dataset, set_use_latex


set_use_latex()

# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 5)


# Compute reconstructions with each different architectures
fig = plt.figure(figsize=(9, 2.7))
ax1 = fig.add_subplot(1, 3, 1)
ax2 = fig.add_subplot(1, 3, 2)
ax3 = fig.add_subplot(1, 3, 3)

# load hyper-parameters
params = Params.load('ellipses_dip')
# params = Params.load('ellipses_diptv')

params.iterations = 15001
reconstructor = DeepImagePriorReconstructor(ray_trafo=dataset.ray_trafo,
                                            hyper_params=params.dict,
                                            callback_func_interval=1,
                                            name='DIP')

for i, (obs, gt) in enumerate([test_data[0], test_data[3], test_data[4]]):
    print('Sample: %d' % i)
    loss_history = []
    psnr_history = []
    reco_history = []
    ssim_history = []
    iter_history = []


    def callback_func(iteration, reconstruction, loss):
        global iter_history, loss_history, psnr_history, ssim_history, reco_history
        if iteration == 0:
            return
        iter_history.append(iteration)
        loss_history.append(loss)
        psnr_history.append(PSNR(reconstruction, gt))
        ssim_history.append(SSIM(reconstruction, gt))
        reco_history.append(reconstruction)


    reconstructor.callback_func = callback_func
    result = reconstructor.reconstruct(obs)

    ax1.plot(iter_history, loss_history)
    ax1.set_yscale('log')
    ax1.set_title('Loss')
    ax1.set_xlabel('Iterations')

    ax2.plot(iter_history, psnr_history)
    ax2.set_title('PSNR')
    ax2.set_xlabel('Iterations')

    ax3.plot(iter_history, ssim_history, label='Sample %d' % i)
    ax3.set_title('SSIM')
    ax3.set_xlabel('Iterations')
    ax3.legend()

plt.tight_layout()
plt.savefig('ellipses_dip_losses.pdf')
plt.savefig('ellipses_dip_losses.pgf')
plt.show()
