from dival import TaskTable, DataPairs
from dival.measure import SSIM, PSNR

from dliplib.reconstructors import diptv_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('lodopab')
test_data = dataset.get_data_pairs('test', 100)

offset = 75
length = 25

obs = list(y for y, x in test_data)
gt = list(x for y, x in test_data)

test_data = DataPairs(obs[offset:offset+length], gt[offset:offset+length], name='test')

# load reconstructor
reconstructor = diptv_reconstructor('lodopab')

# eval on the test-set
print('Eval offset: %d' % offset)
task_table = TaskTable()
task_table.append(reconstructor=reconstructor, measures=[PSNR, SSIM], test_data=test_data)
task_table.run()

print(task_table.results.to_string(show_columns=['misc']))
save_results_table(task_table.results, 'lodopab_diptv_eval__offset_%d' % offset)