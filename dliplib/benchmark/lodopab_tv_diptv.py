from dival import DataPairs

from dliplib.reconstructors import tv_reconstructor, diptv_reconstructor, fbp_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.plot import plot_reconstructors_tests


# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('test', 5)

index = [4]
obs = [test_data[i][0] for i in index]
gt = [test_data[i][1] for i in index]
test_data = DataPairs(obs, gt, name='test')

# load reconstructor
fbp = fbp_reconstructor('lodopab')
tv = tv_reconstructor('lodopab')
diptv = diptv_reconstructor('lodopab')

# compute and plot reconstructions
plot_reconstructors_tests([fbp, tv, diptv],
                          ray_trafo=dataset.ray_trafo,
                          test_data=test_data,
                          save_name='lodopab-tv-diptv4',
                          fig_size=(9, 3),
                          cmap='bone')
