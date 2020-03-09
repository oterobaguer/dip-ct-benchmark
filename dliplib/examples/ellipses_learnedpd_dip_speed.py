import time

from dliplib.reconstructors import learnedpd_dip_reconstructor, diptv_reconstructor
from dliplib.utils.helper import load_standard_dataset, set_use_latex


set_use_latex()

# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 100)
obs, gt = test_data[0]

data_size = 0.001

print('Eval: Learned PD ({}%) + DIP'.format(data_size * 100))
reconstructor = learnedpd_dip_reconstructor('ellipses', data_size)

t_start = time.time()
reco1 = reconstructor.reconstruct(obs)
t_end = time.time()

print('Elapsed time: %d s'% (t_end - t_start))


print('Eval: DIP+TV')
# load hyper-parameters and create reconstructor
reconstructor = diptv_reconstructor('ellipses')

t_start = time.time()
reco2 = reconstructor.reconstruct(obs)
t_end = time.time()

print('Elapsed time: %d s'% (t_end - t_start))