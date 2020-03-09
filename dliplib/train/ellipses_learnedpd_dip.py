import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.reconstructors import learnedpd_dip_reconstructor
from dliplib.utils import Params
from dliplib.utils.helper import select_hyper_best_parameters, load_standard_dataset
from dliplib.utils.reports import save_results_table


dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('validation', 5)

task_table = TaskTable()

data_size = 0.002

# load reconstructor
reconstructor = learnedpd_dip_reconstructor('ellipses', data_size)

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr1': [0.001],
                      'lr2': [1e-6, 1e-7, 1e-8],
                      'scales': [5],
                      # 'gamma': np.logspace(-5, -3, 10),
                      'gamma': [0.0001291549665014884, 0.00021544346900318823, 0.0003162277660],
                      'channels': [(128,) * 5],
                      'skip_channels': [(0, 0, 0, 0, 0)],
                      'initial_iterations': [5000],
                      'iterations': [100, 300, 500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 5000],
                  })

results = task_table.run()

save_results_table(results, 'ellipses_learnedpd_dip_{}'.format(data_size))

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('ellipses_learnedpd_dip_{}'.format(data_size))

