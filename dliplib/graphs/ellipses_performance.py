import matplotlib.pyplot as plt
import numpy as np

from dliplib.utils.helper import set_use_latex


set_use_latex()

sizes = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.10, 0.25, 0.50, 1.00]
ticks = range(len(sizes))

# performance on the different data sizes
learnedpd = {'PSNR': [28.09, 28.45, 29.35, 30.11, 30.84, 31.44, 31.84, 32.15, 32.21, 32.27],
             'SSIM': [0.8621, 0.8778, 0.8997, 0.9124, 0.9258, 0.9282, 0.9360, 0.9367, 0.9390, 0.9403]}

learnedgd = {'PSNR': [27.81, 28.40, 29.15, 29.55, 29.70, 29.84, 29.88, 29.95, 30.07, 30.30],
             'SSIM': [0.8580, 0.8769, 0.8955, 0.9027, 0.9051, 0.9077, 0.9082, 0.9094, 0.9121, 0.9162]}

fbpunet = {'PSNR': [25.42, 25.85, 26.25, 26.77, 27.44, 27.97, 28.49, 28.8, 29.1, 29.36],
           'SSIM': [0.7279, 0.7633, 0.7681, 0.8135, 0.8323, 0.8604, 0.8751, 0.8872, 0.894, 0.8987]}

iradonmap = {'PSNR': [17.83, 18.35, 21.41, 22.64, 23.62, 24.77, 25.61, 26.56, 27.36, 28.02],
             'SSIM': [0.2309, 0.2837, 0.5378, 0.6312, 0.7042, 0.7444, 0.8051, 0.8389, 0.8615, 0.8766]}

fbp = {'PSNR': 24.18,
       'SSIM': 0.5939}

tv = {'PSNR': 27.96,
      'SSIM': 0.8616}

diptv = {'PSNR': 28.94,
         'SSIM': 0.8855}

learnedpd_dip = {'PSNR': [29.23, 29.39, 29.85, 30.39, 30.99, 31.44],
                 'SSIM': [0.8915, 0.8911,  0.904, 0.915, 0.9253, 0.9285]}


fig, ax = plt.subplots(1, 2, figsize=(8, 5.0))
for i, measure in enumerate(['PSNR', 'SSIM']):

    ax[i].axhline(fbp[measure], ticks[0], ticks[-1], label='FBP', color='tab:gray',
                linestyle=':', linewidth=1.5)

    ax[i].axhline(tv[measure], ticks[0], ticks[-1], label='TV', color='tab:orange',
                linestyle='--', linewidth=1.5)

    ax[i].axhline(diptv[measure], ticks[0], ticks[-1], label='DIP+TV', color='tab:brown',
                linestyle='-.', linewidth=1.5)

    ax[i].plot(ticks, iradonmap[measure], label='iRadonMap', color='tab:green',
             linewidth=1.5, marker='o')

    ax[i].plot(ticks, fbpunet[measure], label='FBP+UNet', color='tab:blue',
             linewidth=1.5, marker='o')

    ax[i].plot(ticks, learnedgd[measure], label='LearnedGD', color='tab:red',
             linewidth=1.5, marker='o')

    ax[i].plot(ticks, learnedpd[measure], label='LearnedPD', color='tab:purple',
             linewidth=1.5, marker='o')

    ax[i].plot(ticks[:6], learnedpd_dip[measure], label='LearnedPD + DIP', color='tab:purple',
                linewidth=1.5, marker='o', markerfacecolor='white')

    ax[i].set_xticks(ticks)
    ax[i].set_xticklabels(np.array(sizes) * 100, rotation=45)
    ax[i].set_xlabel('Data size ($\%$)')
    ax[i].set_ylabel(measure)
    ax[i].set_title('Ellipses - Test error')


ax[0].set_ylim([23.6, 32.5])
ax[1].set_ylim([0.57, 0.96])

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
plt.savefig('ellipses-performance.pdf')
plt.show()
