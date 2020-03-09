from dliplib.reconstructors import fbpunet_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 5)

sizes = [0.001, 0.02, 1.00]
reconstructors = []

for size_part in sizes:
    reconstructors.append(fbpunet_reconstructor('ellipses', size_part, pretrained=True))

# compute and plot reconstructions
plot_reconstructors_tests(reconstructors, dataset.ray_trafo, test_data,
                          save_name='ellipses-fbpunet-test',
                          fig_size=(9, 3))
