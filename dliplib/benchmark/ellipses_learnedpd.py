from dival import DataPairs

from dliplib.reconstructors import learnedpd_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 200)

index = [68, 137]

obs = [test_data[i][0] for i in index]
gt = [test_data[i][1] for i in index]
test_data = DataPairs(obs, gt, name='test')

sizes = [0.001, 0.02, 1.00]
reconstructors = []

for size_part in sizes:
    reconstructors.append(learnedpd_reconstructor('ellipses', size_part))

# compute and plot reconstructions
plot_reconstructors_tests(reconstructors, dataset.ray_trafo, test_data,
                          save_name='ellipses-learnedpd-test',
                          fig_size=(9, 3))
