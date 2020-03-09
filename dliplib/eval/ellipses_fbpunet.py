from dival import TaskTable
from dival.measure import PSNR, SSIM

from dliplib.reconstructors import fbpunet_reconstructor
from dliplib.utils.helper import load_standard_dataset
from dliplib.utils.reports import save_results_table


# load data
dataset = load_standard_dataset('ellipses')
test_data = dataset.get_data_pairs('test', 100)
sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]

for size_part in sizes:
    # load reconstructor
    reconstructor = fbpunet_reconstructor('ellipses', size_part, pretrained=True)

    # eval on the test-set
    task_table = TaskTable()
    task_table.append(reconstructor=reconstructor,
                      measures=[PSNR, SSIM],
                      test_data=test_data,
                      options={'skip_training': True})
    task_table.run()

    print(task_table.results.to_string(show_columns=['misc']))
    save_results_table(task_table.results, 'ellipses_fbpunet_eval_100_{}'.format(size_part))
