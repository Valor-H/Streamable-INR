import matplotlib.pyplot as plt
import matplotlib.colors as mc
import numpy as np

## 折线配置
lw_1, lw_2, lw_3, lw_4 = 1, 1.25, 1.5, 1.75
ms_1, ms_2, ms_3, ms_4 = 1, 1.25, 1.5, 2
mk_1, mk_2, mk_3, mk_4 = "o", "^", "*", "D"
pt_1, pt_2, pt_3, pt_4 = ':', '--', '-', '-.'
co_1, co_2, co_3, co_4, co_5, co_6, co_7 = mc.TABLEAU_COLORS['tab:blue'], mc.TABLEAU_COLORS['tab:orange'], mc.TABLEAU_COLORS['tab:green'], mc.TABLEAU_COLORS['tab:red'],  mc.TABLEAU_COLORS['tab:pink'], mc.TABLEAU_COLORS['tab:gray'], mc.TABLEAU_COLORS['tab:olive']
co_8 = mc.BASE_COLORS['c']

fig, ax = plt.subplots(1, 2, figsize=(18, 5))
#############################################
bits = [8, 9, 10, 11, 12, 13, 14, 15, 16]
down_coin = [-12.0594, -9.3345, -6.1331, -2.9393, -0.9760, -0.2519, -0.0709, -0.0186, -0.0055, ]
down_sw = [-11.7090, -8.8343, -6.0366, -3.3188, -1.4115, -0.4333, -0.1210, -0.0303, -0.0091, ]
down_sd = [-10.1390, -7.1818, -4.6734, -2.6073, -1.1643, -0.3957, -0.1143, -0.0301, -0.0077, ]
down_swd = [-12.0786, -8.7374, -5.4907, -2.7881, -1.2039, -0.4410, -0.1326, -0.0382, -0.0093, ]
ax[0].plot(bits, down_coin, pt_3, label="COIN", c=co_1, linewidth=lw_2, marker = mk_1, markersize=ms_4)
ax[0].plot(bits, down_sw, pt_3, label="WSIC", c=co_2, linewidth=lw_2, marker = mk_2, markersize=ms_4)
ax[0].plot(bits, down_sd, pt_3, label="DSIC (ours)", c=co_3, linewidth=lw_2, marker = mk_3, markersize=ms_4)
ax[0].plot(bits, down_swd, pt_3, label="WDSIC (ours)", c=co_4, linewidth=lw_2, marker = mk_4, markersize=ms_4)
ax[0].set_xlabel('number of bits')
ax[0].set_ylabel("PSNR drop [db]")
ax[0].set_xlim(8, 18)
ax[0].set_ylim(-14, 2)
ax[0].grid()
ax[0].legend(loc='lower right')


bpp_11 = [2.515, 1.1, 1.016]
bpp_22 = [2.32, 1.015, 0.774]
bpp_33 = [2.477, 1.084, 0.946]
bpp_44 = [2.43, 1.063, 0.844]
psnr_11 = [30.0414, 29.9476, 29.9476,]
psnr_22 = [28.2912, 28.1700, 28.1700,]
psnr_33 = [29.4851, 29.3241, 29.3241,]
psnr_44 = [29.6873, 29.5545, 29.5545,]
ax[1].plot(bpp_11, psnr_11, pt_3, label="COIN", c=co_2, linewidth=lw_3, )
ax[1].plot(bpp_22, psnr_22, pt_3, label="WSIC", c=co_3, linewidth=lw_2, )
ax[1].plot(bpp_33, psnr_33, pt_3, label="DSIC (ours)", c=co_4, linewidth=lw_3, )
ax[1].plot(bpp_44, psnr_44, pt_3, label="WDSIC (ours)", c=co_1, linewidth=lw_3, )
ax[1].plot([2.515, 2.32, 2.477, 2.43], [30.0414, 28.2912, 29.4851, 29.6873], '*', label="basic", c=co_1, linewidth=lw_3, marker = mk_1, markersize=8)
ax[1].plot([1.1, 1.015, 1.084, 1.063], [29.9476, 28.1700, 29.3241, 29.5545], '*', label="+ quantization", c=co_2, linewidth=lw_3, marker = mk_2, markersize=8)
ax[1].plot([1.016, 0.774, 0.946, 0.844], [29.9476, 28.1700, 29.3241, 29.5545], '*', label="+ entropy coding", c=co_3, linewidth=lw_3, marker = mk_4, markersize=8)

ax[1].set_xlabel('Bit-rate [bpp]')
ax[1].set_ylabel("PSNR [db]")
ax[1].set_xlim(0.4, 3.3)
ax[1].set_ylim(27.3, 30.3)
ax[1].grid()
ax[1].legend(loc='lower right')

fig.savefig(f"./plot_imgs/quant_entropy.jpg", dpi=300, bbox_inches='tight')
print("finish !!!")
plt.clf()
plt.close()