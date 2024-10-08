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

fig, ax = plt.subplots(1, 2, figsize=(14, 4))


bpp_coin = [0.138, 0.272, 0.458, 0.705, 1.016]
bpp_sw = [0.179, 0.271, 0.376, 0.496, 0.626, 0.774]
bpp_sd = [0.181, 0.346, 0.503, 0.654, 0.801, 0.946]
bpp_swd = [0.195, 0.372, 0.54, 0.68, 0.741, 0.844]
psnr_coin = [24.3635, 25.9738, 27.4592, 28.9281, 29.9476,]
psnr_sw = [25.2997, 25.9414, 26.5317, 27.0954, 27.6534, 28.1700,]
psnr_sd = [24.2507, 26.9462, 28.2138, 28.9186, 29.2621, 29.3241,]
psnr_swd = [24.3210, 27.0898, 28.3665, 29.0650, 29.3032, 29.5545, ]
bpp_jpeg = [0.22115325927734372,
      0.32661437988281244,
      0.42312622070312506,
      0.5083855523003472,
      0.5878660413953993,
      0.6601265801323783,
      0.728946261935764,
      0.786026848687066,
      0.8497187296549479,
      0.9060007731119791,
      0.9643800523546006,
      1.0372119479709203,
      1.1273964775933163,
      1.23984612358941,
      1.3688269721137154,
      1.5718070136176217,
      1.8588163587782118,
      2.350199381510416,
      3.4013400607638893]
bpp_jpeg2000 = [2.3973227606879344,
      1.198240492078993,
      0.7982796563042536,
      0.5988286336263021,
      0.47928958468967026,
      0.3986248440212674,
      0.3418426513671875,
      0.2985627916124132,
      0.2657267252604167,
      0.239408704969618,
      0.21756574842664925,
      0.19917382134331593,
      0.1844355265299479,
      0.17067379421657983,
      0.1594840155707465,
      0.14888424343532988,
      0.1407623291015625,
      0.1324556138780382,
      0.12579854329427081]
psnr_jpeg = [23.779894921045457,
      26.57723034342358,
      28.042246379237767,
      29.04180914810682,
      29.78473021842612,
      30.378313496658652,
      30.903164761925012,
      31.307827225129476,
      31.70484530103775,
      32.05865273799422,
      32.39929573599792,
      32.789915599755076,
      33.234742421489976,
      33.792594320368266,
      34.39429509119509,
      35.239001425077355,
      36.32886455167303,
      37.9121247026983,
      40.556657112988766]
psnr_jpeg2000 = [39.792000990389106,
      35.680719121855866,
      33.4899612266554,
      32.08974009483378,
      31.07457563807384,
      30.30076938092785,
      29.696116673252778,
      29.191735537599026,
      28.774226514669053,
      28.416590303662094,
      28.099224033143784,
      27.796635846897356,
      27.554469623734875,
      27.316993364478737,
      27.11825064017241,
      26.922846006615767,
      26.757424438699193,
      26.586234996125643,
      26.440382666052425]
ax[0].plot(bpp_jpeg, psnr_jpeg, pt_3, label="JPEG", c=co_5, linewidth=lw_1)
ax[0].plot(bpp_jpeg2000, psnr_jpeg2000, pt_3, label="JPEG2000", c=co_6, linewidth=lw_1)
ax[0].plot(bpp_coin, psnr_coin, pt_3, label="COIN", c=co_2, linewidth=lw_1, marker = mk_1, markersize=2)
ax[0].plot(bpp_sw, psnr_sw, pt_3, label="WSIC", c=co_3, linewidth=lw_1, marker = mk_1, markersize=2)
ax[0].plot(bpp_sd, psnr_sd, pt_3, label="DSIC (ours)", c=co_4, linewidth=lw_3, marker = mk_1, markersize=3)
ax[0].plot(bpp_swd, psnr_swd, pt_3, label="WDSIC (ours)", c=co_1, linewidth=lw_3, marker = mk_1, markersize=3)
ax[0].set_xlabel('Bit-rate [bpp]')
ax[0].set_ylabel("PSNR [db]")
ax[0].set_xlim(0.1, 1.1)
ax[0].set_ylim(23, 34)
ax[0].grid()
ax[0].legend(loc='lower right')


bpp_coin = [0.138, 0.272, 0.458, 0.705, 1.016]
bpp_sw = [0.179, 0.271, 0.376, 0.496, 0.626, 0.774]
bpp_sd = [0.181, 0.346, 0.503, 0.654, 0.801, 0.946]
bpp_swd = [0.195, 0.372, 0.54, 0.68, 0.741, 0.844]
ssim_coin = [0.5989, 0.6597, 0.7096, 0.7657, 0.8010,]
ssim_sw = [0.6328, 0.6539, 0.6739, 0.6972, 0.7180, 0.7376,]
ssim_sd = [0.5974, 0.6913, 0.7340, 0.7581, 0.7697, 0.7735,]
ssim_swd = [0.5996, 0.6992, 0.7423, 0.7649, 0.7704, 0.7793,]
bpp_jpeg = [0.22115325927734372,
      0.32661437988281244,
      0.42312622070312506,
      0.5083855523003472,
      0.5878660413953993,
      0.6601265801323783,
      0.728946261935764,
      0.786026848687066,
      0.8497187296549479,
      0.9060007731119791,
      0.9643800523546006,
      1.0372119479709203,
      1.1273964775933163,
      1.23984612358941,
      1.3688269721137154,
      1.5718070136176217,
      1.8588163587782118,
      2.350199381510416,
      3.4013400607638893]
bpp_jpeg2000 = [2.3973227606879344,
      1.198240492078993,
      0.7982796563042536,
      0.5988286336263021,
      0.47928958468967026,
      0.3986248440212674,
      0.3418426513671875,
      0.2985627916124132,
      0.2657267252604167,
      0.239408704969618,
      0.21756574842664925,
      0.19917382134331593,
      0.1844355265299479,
      0.17067379421657983,
      0.1594840155707465,
      0.14888424343532988,
      0.1407623291015625,
      0.1324556138780382,
      0.12579854329427081]
ssim_jpeg = [0.8030709524949392,
      0.8898622319102287,
      0.9235836813847224,
      0.9427085692683855,
      0.9533494859933853,
      0.9606156249841055,
      0.9661434814333916,
      0.9696919098496437,
      0.9726496761043867,
      0.9752530703941981,
      0.9772666270534197,
      0.9792757481336594,
      0.9813638602693876,
      0.9837072988351186,
      0.9857503150900205,
      0.9881909117102623,
      0.9905517573157946,
      0.9929814984401067,
      0.9956368828813235]
ssim_jpeg2000 = [0.9915901000301043,
      0.9807158832748731,
      0.970682812233766,
      0.9609081571300825,
      0.9530079588294029,
      0.9446083009243011,
      0.9375656371315321,
      0.9304135168592135,
      0.9243723601102829,
      0.919123962521553,
      0.9140403121709824,
      0.9093747685352961,
      0.904474933942159,
      0.8996987566351891,
      0.8954376926024755,
      0.889873224000136,
      0.885848231613636,
      0.88182615985473,
      0.8782255152861277]
ax[1].plot(bpp_jpeg, ssim_jpeg, pt_3, label="JPEG", c=co_5, linewidth=lw_1)
ax[1].plot(bpp_jpeg2000, ssim_jpeg2000, pt_3, label="JPEG2000", c=co_6, linewidth=lw_1)
ax[1].plot(bpp_coin, ssim_coin, pt_3, label="COIN", c=co_2, linewidth=lw_1, marker = mk_1, markersize=2)
ax[1].plot(bpp_sw, ssim_sw, pt_3, label="WSIC", c=co_3, linewidth=lw_1, marker = mk_1, markersize=2)
ax[1].plot(bpp_sd, ssim_sd, pt_3, label="DSIC (ours)", c=co_4, linewidth=lw_3, marker = mk_1, markersize=3)
ax[1].plot(bpp_swd, ssim_swd, pt_3, label="WDSIC (ours)", c=co_1, linewidth=lw_3, marker = mk_1, markersize=3)

ax[1].set_xlabel('Bits per pixel')
ax[1].set_ylabel("MS-SSIM")
ax[1].set_xlim(0.1, 1.1)
ax[1].set_ylim(0.5, 1.0)
ax[1].grid()
ax[1].legend(loc='lower right')


fig.savefig(f"./plot/plot_imgs/streamable_result.jpg", dpi=300, bbox_inches='tight')
print("finish streamable_result.jpg!!!")
plt.clf()
plt.close()