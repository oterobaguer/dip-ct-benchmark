import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.reconstructors import learnedpd_dip_reconstructor
from dliplib.utils import Params
from dliplib.utils.data.datasets import CachedDataset
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table


dataset = load_standard_dataset('lodopab_200', ordered=True)
test_data = dataset.get_data_pairs('train', 5)

task_table = TaskTable()
size_part = 0.0001

# load reconstructor
reconstructor = learnedpd_dip_reconstructor('lodopab_200', size_part)


# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr1': [0.0005],
                      'lr2': [1e-6],
                      'scales': [6],
                      'gamma': [2.5, 3.0, 4.0],
                      'channels': [(128,) * 6],
                      'skip_channels': [(0, 0, 0, 0, 4, 4)],
                      'initial_iterations': [5000, 9750],
                      'iterations': [200, 250, 300, 350, 400, 500],
                      "loss_function": ["poisson"]
})

results = task_table.run()

save_results_table(results, 'lodopab_200_learnedpd_dip_{}'.format(size_part))

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('lodopab_200_learnedpd_dip_{}'.format(size_part))
