import numpy as np

wbase, dbase = 30, 4
wgrow, dgrow = 5, 2
w, d = [], []
for i in range(5):
    w.append(wbase + i * wgrow)
    d.append(dbase + i * dgrow)
# ignore above set
# w, d = [28, 36, 40, 48, 40, 49], [6, 6, 9, 9, 12, 12]
sn = np.arange(1, len(w)+1)
pix_num = 512 * 768
bit_num = 32

## 计算bpp
coin_bpp, sw_bpp, sd_bpp, swd_bpp = [], [], [], []
for i in range(len(w)):
    coin_num = w[i]*(6+d[i]) + d[i]*w[i]*w[i] + 3
    sw_num = w[i]*(6+d[i]) + d[i]*w[i]*w[i] + 3 - (sn[i]-1)*d[i]*wgrow*wbase - (sn[i]-2)*(sn[i]-1)*d[i]*wgrow*wgrow/2
    sd_num = w[i]*(3+d[i]) + d[i]*w[i]*w[i] + 3 + 3*sn[i]*w[i]
    swd_num = w[i]*(3+d[i]) + d[i]*w[i]*w[i] + 3 - (sn[i]-1)*d[i]*wgrow*wbase - (sn[i]-2)*(sn[i]-1)*d[i]*wgrow*wgrow/2 + 3*sn[i]*w[i]                        
    coin_bpp.append(round(coin_num * bit_num / pix_num, 3))
    sw_bpp.append(round(sw_num * bit_num / pix_num, 3))
    sd_bpp.append(round(sd_num * bit_num / pix_num, 3))
    swd_bpp.append(round(swd_num * bit_num / pix_num, 3))
print("COIN: W", w, "D", d, "BPP", coin_bpp)
print("SW: W", w, "D", d, "BPP", sw_bpp)
print("SD: W", w, "D", d, "BPP", sd_bpp)
print("SWD: W", w, "D", d, "BPP", swd_bpp)