import numpy as np
from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.utils import Params
from dliplib.utils.helper import select_hyper_best_parameters
from dliplib.utils.helper import load_standard_dataset
from dliplib.reconstructors.dip import DeepImagePriorReconstructor
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('lodopab_200')
test_data = dataset.get_data_pairs('validation', 4)

task_table = TaskTable()

# create the reconstructor
reconstructor = DeepImagePriorReconstructor(dataset.ray_trafo)

# create a Dival task table and run it
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM],
                  test_data=test_data,
                  hyper_param_choices={
                      'lr': [0.0007],
                      'scales': [6],
                      'gamma': np.linspace(2, 7, 10),
                      'channels': [(128,) * 6],
                      'skip_channels': [(0, 0, 0, 0, 4, 4)],
                      'iterations': [2000, 3000, 4000, 5000, 6000, 7000, 8000, 9000, 10000],
                      'loss_function': ['poisson']
})

results = task_table.run()

save_results_table(results, 'lodopab_200_diptv')

# select the best hyper-parameters and save them
best_choice, best_error = select_hyper_best_parameters(results)

print(results.to_string(show_columns=['misc']))
print(best_choice)
print(best_error)

params = Params(best_choice)
params.save('lodopab_200_diptv')
