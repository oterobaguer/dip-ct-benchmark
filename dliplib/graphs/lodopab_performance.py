import matplotlib.pyplot as plt
import numpy as np

from dliplib.utils.helper import set_use_latex


plt.style.use('seaborn-whitegrid')
set_use_latex()

sizes = [0.0001, 0.001, 0.01, 0.10, 1.00]
ticks = range(len(sizes))

# performance on the different data sizes
# psnr: 34.67, ssim: 0.8171
# psnr: 34.55, ssim: 0.815

learnedgd = {'PSNR [db]': [32.70, 33.81, 34.29, 34.67, 34.67],
             'SSIM': [0.7860, 0.8043, 0.8103, 0.8171, 0.8171]}

learnedpd = {'PSNR [db]': [32.68, 34.65, 35.27, 35.63, 35.73],
             'SSIM': [0.7842, 0.8227, 0.8303, 0.8401, 0.8426]}

fbpunet = {'PSNR [db]': [31.36, 33.27, 34.62, 35.18, 35.48],
           'SSIM': [0.7727, 0.7982, 0.8209, 0.8313, 0.8371]}

iradonmap = {'PSNR [db]': [14.82, 17.67, 22.73, 28.69, 30.99],
             'SSIM': [0.3737, 0.4438, 0.5361, 0.6929, 0.7486]}

tv = {'PSNR [db]': 32.95,
      'SSIM': 0.8034}

fbp = {'PSNR [db]': 30.37,
       'SSIM': 0.7386}

diptv = {'PSNR [db]': 34.4425,
         'SSIM': 0.8143}

learnedpd_dip = {'PSNR [db]': [0, 0, 0, 0, 0],
                 'SSIM': [0, 0, 0, 0, 0]}

fig, ax = plt.subplots(1, 2, figsize=(8, 4.0))
for i, measure in enumerate(['PSNR [db]', 'SSIM']):

    ax[i].axhline(fbp[measure], ticks[0], ticks[-1], label='FBP', color='tab:gray',
                  linestyle=':', linewidth=1.5)

    ax[i].axhline(tv[measure], ticks[0], ticks[-1], label='TV', color='tab:orange',
                  linestyle='--', linewidth=1.5)

    ax[i].axhline(diptv[measure], ticks[0], ticks[-1], label='DIP+TV', color='tab:brown',
                  linestyle='-.', linewidth=1.5)

    ax[i].plot(ticks, iradonmap[measure], label='iRadonMap', color='tab:green',
               linewidth=1.5, marker='o')

    ax[i].plot(ticks, fbpunet[measure], label='FBP+U-Net', color='tab:blue',
               linewidth=1.5, marker='o')

    ax[i].plot(ticks, learnedgd[measure], label='LearnedGD', color='tab:red',
               linewidth=1.5, marker='o')

    ax[i].plot(ticks, learnedpd[measure], label='LearnedPD', color='tab:purple',
               linewidth=1.5, marker='o')

    # ax[i].plot(ticks[:6], learnedpd_dip[measure], label='LearnedPD + DIP', color='tab:purple',
    #             linewidth=1.5, marker='o', markerfacecolor='white')

    ax[i].set_xticks(ticks)
    ax[i].set_xticklabels(np.array(sizes) * 100, rotation=45)
    ax[i].set_xlabel('Data size [$\%$]')
    ax[i].set_ylabel(measure)
    ax[i].set_title('LoDoPaB - Test error')


ax[0].set_ylim([26.0, 37.0])
ax[1].set_ylim([0.68, 0.86])

for i in range(2):
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width, box.height * 0.6])

h, l = ax[0].get_legend_handles_labels()
ax[0].legend([h[3], h[4], h[5], h[6]], [l[3], l[4], l[5], l[6]], bbox_to_anchor=(0.0, -0.45, 1., 0.5), loc=3,
             ncol=2, mode="expand", frameon=False)
h, l = ax[1].get_legend_handles_labels()
ax[1].legend([h[2], h[0], h[1]], [l[2], l[0], l[1]], bbox_to_anchor=(0.0, -0.45, 1., 0.5), loc=3,
             ncol=2, mode="expand", frameon=False)

plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.savefig('lodopab-performance.pdf')
plt.show()
