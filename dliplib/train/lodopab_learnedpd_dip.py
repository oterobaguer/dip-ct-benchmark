import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.reconstructors import learnedpd_dip_reconstructor
from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table


dataset = load_standard_dataset('lodopab', ordered=True)
ray_trafo = dataset.ray_trafo

test_data = dataset.get_data_pairs('train', 3)

task_table = TaskTable()

size_part = 0.0001

# load reconstructor
reconstructor = learnedpd_dip_reconstructor('lodopab', size_part)


# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr1': [0.001],
                      'lr2': [1e-5],
                      'scales': [6],
                      'gamma': [4.0],
                      'channels': [(128,) * 6],
                      'skip_channels': [(0, 0, 4, 4, 4, 4), (0, 0, 0, 0, 4, 4)],
                      'initial_iterations': [4000, 5000],
                      'iterations': [4000, 5000],
                      "loss_function": ["poisson"]
})

results = task_table.run()

save_results_table(results, 'lodopab_learnedpd_dip_{}'.format(size_part))

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('lodopab_learnedpd_dip_{}'.format(size_part))
