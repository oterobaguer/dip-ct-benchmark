from dival import DataPairs

from dliplib.reconstructors import learnedpd_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('lodopab', ordered=False)
test_data = dataset.get_data_pairs('test', 1000)


sizes = [0.0001, 0.01, 1.00]
reconstructors = []

for size_part in sizes:
    reconstructors.append(learnedpd_reconstructor('lodopab', size_part, pretrained=True))

for i in [0, 551]:
    obs, gt = test_data[i]
    test_data = DataPairs([obs], [gt], name='test')

    # compute and plot reconstructions
    plot_reconstructors_tests(reconstructors, dataset.ray_trafo, test_data,
                              save_name='lodopab-learnedpd-test-%d' % i,
                              fig_size=(9, 3),
                              cmap='bone')
