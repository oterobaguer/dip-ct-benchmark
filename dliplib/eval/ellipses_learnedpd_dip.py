from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.reconstructors import learnedpd_dip_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 100)

data_size = 0.001

# load reconstructor
reconstructor = learnedpd_dip_reconstructor('ellipses', data_size)

# eval on the test-set
task_table = TaskTable()
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM], test_data=test_data)
task_table.run()

print(task_table.results.to_string(show_columns=['misc']))
save_results_table(task_table.results, 'ellipses_learnedpd_dip_eval_{}'.format(data_size))





