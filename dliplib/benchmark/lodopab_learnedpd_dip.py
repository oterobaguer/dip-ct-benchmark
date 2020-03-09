from dival import DataPairs

from dliplib.reconstructors import learnedpd_reconstructor, fbp_reconstructor, tv_reconstructor, \
    learnedpd_dip_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('test', 5)

# index = [0, 2, 68]
index = [0, 3]
obs = [test_data[i][0] for i in index]
gt = [test_data[i][1] for i in index]
test_data = DataPairs(obs, gt, name='test')

data_size = 0.001

# load reconstructors
fbp = fbp_reconstructor('lodopab')
tv = tv_reconstructor('lodopab')
learnedpd = learnedpd_reconstructor('lodopab', data_size)
learnedpd_dip = learnedpd_dip_reconstructor('lodopab', data_size)

# compute example reconstructions
plot_reconstructors_tests([fbp, learnedpd, learnedpd_dip],
                          dataset.ray_trafo,
                                     test_data[:5],
                          save_name='lodopab-learnedpd-dip-{}'.format(data_size),
                          fig_size=(9, 3))