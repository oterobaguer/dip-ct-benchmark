import matplotlib.pyplot as plt
import numpy as np

from dliplib.utils.helper import set_use_latex


plt.style.use('seaborn-whitegrid')
set_use_latex()

sizes = [0.0001, 0.001, 0.01, 0.10, 1.00]
ticks = range(len(sizes))

# performance on the different data sizes
learnedgd = {'PSNR [db]': [29.87, 31.28, 31.83, 32.7, 32.7],
             'SSIM': [0.7151, 0.7473, 0.7602, 0.7802, 0.7802]}

learnedpd = {'PSNR [db]': [29.65, 32.48, 33.21, 33.53, 33.64],
             'SSIM': [0.7343, 0.7771, 0.7929, 0.799, 0.8020]}

fbpunet = {'PSNR [db]': [29.33, 31.58, 32.6, 33.19, 33.55],
           'SSIM': [0.7143, 0.7616, 0.7818, 0.7931, 0.7994]}

iradonmap = {'PSNR [db]': [14.61, 18.77, 24.63, 31.27, 32.45],
             'SSIM': [0.3529, 0.4492, 0.6031, 0.7569, 0.7781]}

tv = {'PSNR [db]': 30.89,
      'SSIM': 0.7563}

# psnr: 28.38, ssim: 0.6492
fbp = {'PSNR [db]': 28.38,
       'SSIM': 0.6492}


diptv = {'PSNR [db]': 32.51,
         'SSIM': 0.7803}

learnedpd_dip = {'PSNR [db]': [32.52, 32.78, 33.21],
                 'SSIM': [0.7822, 0.7821, 0.7929]}

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

    ax[i].plot(ticks[:3], learnedpd_dip[measure], label='LearnedPD + DIP', color='tab:purple',
               linewidth=1.5, marker='o', markerfacecolor='white')

    ax[i].set_xticks(ticks)
    ax[i].set_xticklabels(np.array(sizes) * 100, rotation=45)
    ax[i].set_xlabel('Data size [$\%$]')
    ax[i].set_ylabel(measure)
    ax[i].set_title('LoDoPaB (200) - Test error')


ax[0].set_ylim([24.0, 35.0])
ax[1].set_ylim([0.58, 0.82])

for i in range(2):
    box = ax[i].get_position()
    ax[i].set_position([box.x0, box.y0, box.width, box.height * 0.6])

h, l = ax[0].get_legend_handles_labels()
ax[0].legend([h[3], h[4], h[5], h[6]], [l[3], l[4], l[5], l[6]], bbox_to_anchor=(0.0, -0.45, 1., 0.5), loc=3,
             ncol=2, mode="expand", frameon=False)
h, l = ax[1].get_legend_handles_labels()
ax[1].legend([h[2], h[7], h[0], h[1]], [l[2], l[7], l[0], l[1]], bbox_to_anchor=(0.0, -0.45, 1., 0.5), loc=3,
             ncol=2, mode="expand", frameon=False)

plt.tight_layout()
plt.tight_layout()
plt.tight_layout()
plt.savefig('lodopab-200-performance.pdf')
plt.show()
