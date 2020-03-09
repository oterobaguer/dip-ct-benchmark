from dival import TaskTable
from dival.measure import SSIM, PSNR

from dliplib.reconstructors import dip_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('test', 10)

# load reconstructor
reconstructor = dip_reconstructor('lodopab')

# eval on the test-set
task_table = TaskTable()
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM], test_data=test_data)
task_table.run()

print(task_table.results.to_string(show_columns=['misc']))
save_results_table(task_table.results, 'lodopab_dip_eval')