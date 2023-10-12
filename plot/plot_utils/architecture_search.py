import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

lw_1, lw_2, lw_3, lw_4 = 1, 1.25, 1.5, 1.75
ms_1, ms_2, ms_3, ms_4 = 1, 1.25, 1.5, 2
mk_1, mk_2, mk_3, mk_4 = "o", "v", "*", "D"
pt_1, pt_2, pt_3, pt_4 = ':', '--', '-', '-.'
co_1, co_2, co_3, co_4,  = mc.TABLEAU_COLORS['tab:blue'], mc.TABLEAU_COLORS['tab:orange'], mc.TABLEAU_COLORS['tab:green'], mc.TABLEAU_COLORS['tab:red']
co_5, co_6, co_7, co_8 = mc.TABLEAU_COLORS['tab:pink'], mc.TABLEAU_COLORS['tab:gray'], mc.TABLEAU_COLORS['tab:olive'], mc.TABLEAU_COLORS['tab:purple']

## coin
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
# width_1 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# psnr_1 = [29.6292, 30.3676, 30.7091, 31.2539, 31.4499, 31.2200, 31.8236, 31.3871, 31.7005, 32.2110, 32.1917, 32.0808, 31.5215]
# width_2 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14]
# psnr_2 = [28.6357, 28.9932, 29.6268, 29.2976, 29.4223, 29.6283, 30.0420, 30.1576, 29.9548, 29.7546, 29.7623, 30.0046, 29.3144]
# width_3 = [2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
# psnr_3 = [26.8451, 27.7701, 28.0665, 28.0800, 28.0925, 28.7733, 28.6313, 28.7086, 28.4790, 27.8624, 27.8318]
# width_4 = [2, 3, 4, 5, 6, 7, 8, 9, 10]
# psnr_4 = [25.4325, 25.8884, 26.5175, 26.8200, 27.3222, 27.3309, 27.0505, 27.2828, 26.6477]
# width_5 = [2, 3, 4, 5, 6, 7, 8]
# psnr_5 = [23.7804, 24.4349, 25.0913, 24.8278, 24.7962, 24.8340, 24.6174,]
# ax[0].plot(width_1, psnr_1, pt_3, label="bpp=2.4", c=co_1, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[0].plot(width_2, psnr_2, pt_3, label="bpp=1.2", c=co_2, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[0].plot(width_3, psnr_3, pt_3, label="bpp=0.6", c=co_3, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[0].plot(width_4, psnr_4, pt_3, label="bpp=0.3", c=co_4, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[0].plot(width_5, psnr_5, pt_3, label="bpp=0.15", c=co_5, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[0].set_ylim(21.5, 33.5)
# ax[0].set_xlabel('Number of hidden layers')
# ax[0].set_ylabel("PSNR [db]")
# ax[0].grid()
# ax[0].legend(loc='lower right')

## sw
width_1 = [0.406, 0.656, 0.977, 1.37, 1.835, 2.372]
psnr_1 = [26.0156, 26.5779, 26.9395, 27.5033, 28.0658, 28.5697,]
width_2 = [0.42, 0.669, 0.989, 1.379, 1.84, 2.371]
psnr_2 = [27.1250, 27.6887, 28.2108, 28.7117, 29.1526, 29.6120,]
width_3 = [0.404, 0.656, 0.98, 1.378, 1.85, 2.394]
psnr_3 = [27.4002, 27.8979, 28.4729, 28.9424, 29.4119, 29.8912,]
width_4 = [0.419, 0.663, 0.977, 1.358, 1.809, 2.329]
psnr_4 = [27.5949, 28.4354, 28.9916, 29.4299, 29.8485, 30.2874,]
width_5 = [0.41, 0.656, 0.973, 1.36, 1.817, 2.344]
psnr_5 = [27.7072, 28.3124, 28.8108, 29.3632, 29.8186, 30.4026,]
width_6 = [0.413, 0.656, 0.969, 1.35, 1.8, 2.32]
psnr_6 = [28.1001, 28.6523, 29.3559, 29.9168, 30.4569, 30.9030,]
width_7 = [0.403, 0.666, 1.008, 1.428, 1.928, 2.506]
psnr_7 = [27.3047, 28.0986, 28.9030, 29.5390, 30.0847, 30.6594,]
width_8 = [0.416, 0.67, 0.997, 1.397, 1.871, 2.418]       ## W [14, 30, 46, 62, 78] D [8, 8, 8, 8, 8]
psnr_8 = [27.5967, 28.3657, 28.9397, 29.5458, 30.0578, 30.6049,]
ax[0].plot(width_1, psnr_1, pt_3, label="depth = 2", c=co_7, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_2, psnr_2, pt_3, label="depth = 3", c=co_2, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_3, psnr_3, pt_3, label="depth = 4", c=co_3, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_4, psnr_4, pt_3, label="depth = 5", c=co_4, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_5, psnr_5, pt_3, label="depth = 6", c=co_5, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_6, psnr_6, pt_3, label="depth = 7", c=co_1, linewidth=lw_3, marker = mk_1, markersize=ms_4)
ax[0].plot(width_7, psnr_7, pt_3, label="depth = 8", c=co_6, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].plot(width_8, psnr_8, pt_3, label="depth = 9", c=co_8, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[0].set_ylim(24.5, 32)
ax[0].set_ylabel("PSNR [db]")
ax[0].set_xlabel('Bit-rate [bpp]')
ax[0].grid()
ax[0].legend(loc='lower right')


## sd
width_1 = [0.287, 0.83, 1.374, 1.918, 2.461]
psnr_1 = [25.4325, 29.7476, 31.1853, 31.6472, 31.6604,]
width_2 = [0.359, 0.875, 1.392, 1.908, 2.425]
psnr_2 = [25.6580, 29.7637, 31.3553, 31.9511, 32.2386,]
width_3 = [0.423, 0.814, 1.244, 1.655, 2.066, 2.477]
psnr_3 = [26.4911, 29.8707, 31.0696, 31.8344, 32.3259, 32.5765]
width_4 = [0.492, 0.971, 1.449, 1.928, 2.407]
psnr_4 = [26.8864, 29.8994, 31.2556, 32.1890, 32.5240,]
ax[1].plot(width_1, psnr_1, pt_3, label="width = 40", c=co_3, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[1].plot(width_2, psnr_2, pt_3, label="width = 45", c=co_2, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[1].plot(width_3, psnr_3, pt_3, label="width = 49", c=co_1, linewidth=lw_3, marker = mk_1, markersize=ms_4)
ax[1].plot(width_4, psnr_4, pt_3, label="width = 53", c=co_4, linewidth=lw_1, marker = mk_1, markersize=ms_4)
ax[1].set_ylim(24, 34)
ax[1].set_ylabel("PSNR [db]")
ax[1].set_xlabel('Bit-rate [bpp]')
ax[1].grid()
ax[1].legend(loc='lower right')


# ## swd
# width_1 = [0.166, 0.413, 0.834, 1.475, 2.385]
# psnr_1 = [24.6189, 27.3951, 28.7997, 30.1615, 31.4053]
# width_2 = [0.166, 0.449, 0.892, 1.537, 2.421]
# psnr_2 = [24.6189, 27.8164, 29.5099, 30.6549, 31.7030]
# width_3 = [0.166, 0.495, 0.959, 1.584, 2.397]
# psnr_3 = [24.6189, 27.7814, 29.1885, 30.0948, 30.8848]
# width_4 = [0.166, 0.548, 1.035, 1.644, 2.391]
# psnr_4 = [24.6189, 28.1133, 29.4657, 30.1853, 30.8800]
# ax[3].plot(width_1, psnr_1, pt_3, label="max depth = 6", c=co_5, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[3].plot(width_2, psnr_2, pt_3, label="max depth = 10", c=co_1, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[3].plot(width_3, psnr_3, pt_3, label="max depth = 14", c=co_2, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[3].plot(width_4, psnr_4, pt_3, label="max depth = 18", c=co_4, linewidth=lw_1, marker = mk_1, markersize=ms_4)
# ax[3].set_ylim(23, 33)
# ax[3].set_ylabel("PSNR [db]")
# ax[3].set_xlabel('Bit-rate [bpp]')
# ax[3].grid()
# ax[3].legend(loc='lower right')


fig.savefig(f"./plot_imgs/architectural_search.jpg", dpi=300, bbox_inches='tight')
print("finish !!!")
plt.clf()
plt.close()