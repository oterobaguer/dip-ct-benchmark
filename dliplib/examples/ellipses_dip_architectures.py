import matplotlib.pyplot as plt

from dival.util.plot import plot_image
from dival.measure import PSNR, SSIM
from dival import DataPairs

from dliplib.utils import Params
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.utils.helper import load_standard_dataset, set_use_latex


set_use_latex()

# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 5)

index = 3
obs, gt = test_data[index]
test_data = DataPairs([obs], [gt], name='test: %d' % index)

# forward operator
ray_trafo = dataset.ray_trafo
params = Params.load('ellipses_dip')

results = []

# Architecture-hyper-parameters
channels = [1, 8, 32, 64, 128, 128, 128, 128]
scales = [5, 5, 5, 5, 5, 4, 3, 2]
iters = [5000] * 8

# Compute reconstructions with each different architectures
for ch, sc, it in zip(channels, scales, iters):
    print('Channels: %d, Scales: %d' % (ch, sc))
    params.channels = (ch,) * sc
    params.skip_channels = (0,) * sc
    params.scales = sc
    params.iterations = it
    reconstructor = DeepImagePriorReconstructor(ray_trafo=ray_trafo, hyper_params=params.dict, name='DIP')
    result = reconstructor.reconstruct(obs)
    results.append(result)

ch = 128
sc = 5
skip_channels = [[0, 0, 0, 0, 4],
                 [0, 0, 0, 4, 4],
                 [0, 0, 4, 4, 4],
                 [0, 4, 4, 4, 4]]

for skip in skip_channels:
    print('Skip Channels:')
    print(skip)
    params.channels = (ch,) * sc
    params.skip_channels = skip
    params.scales = sc
    reconstructor = DeepImagePriorReconstructor(ray_trafo=ray_trafo, hyper_params=params.dict, name='DIP')
    result = reconstructor.reconstruct(obs)
    results.append(result)

fig = plt.figure(figsize=(9.1, 8.3))
for i in range(len(results)):
    ax = fig.add_subplot(3, 4, i+1)
    psnr = PSNR(results[i], gt)
    ssim = SSIM(results[i], gt)
    plot_image(results[i], ax=ax, xticks=[], yticks=[], cmap='pink')
    if i < 8:
        ax.set_title('Channels: %d, Scales: %d' % (channels[i], scales[i]))
    else:
        ax.set_title('Skip: {}'.format(skip_channels[i-8]))
    ax.set_xlabel('PSNR: %.2f, SSIM: %.4f' % (psnr, ssim))

plt.tight_layout()
plt.tight_layout()

plt.savefig('ellipses_architectures.pdf')
plt.savefig('ellipses_architectures.pgf')
plt.show()
