from dival import DataPairs

from dliplib.reconstructors import learnedpd_reconstructor, fbp_reconstructor, tv_reconstructor, \
    learnedpd_dip_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 200)

# index = [0, 2, 68]
index = [68, 137]
obs = [test_data[i][0] for i in index]
gt = [test_data[i][1] for i in index]
test_data = DataPairs(obs, gt, name='test')

data_size = 0.002

# load reconstructors
fbp = fbp_reconstructor('ellipses')
tv = tv_reconstructor('ellipses')
learnedpd = learnedpd_reconstructor('ellipses', data_size)
learnedpd_dip = learnedpd_dip_reconstructor('ellipses', data_size)

# compute example reconstructions
plot_reconstructors_tests([fbp, learnedpd, learnedpd_dip],
                          dataset.ray_trafo,
                                     test_data[:5],
                          save_name='ellipses-learnedpd-dip-{}'.format(data_size),
                          fig_size=(9, 3))