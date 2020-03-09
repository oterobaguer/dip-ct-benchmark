from dliplib.reconstructors import learnedgd_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests

# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('test', 5)

sizes = [0.001, 0.02, 1.00]
reconstructors = []

for size_part in sizes:
    reconstructors.append(learnedgd_reconstructor('lodopab', size_part, pretrained=True))

# compute and plot reconstructions
plot_reconstructors_tests(reconstructors, dataset.ray_trafo, test_data,
                          save_name='lodopab-learnedgd-test',
                          fig_size=(9, 3),
                          cmap='bone')
