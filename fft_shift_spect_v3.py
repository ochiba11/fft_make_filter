import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import cv2
import os
from PIL import Image

#画像imgの引数idxの4方枠組みのみの画素値を合計する
#pixel数で正規化(除算)し戻り値とする
def getOutPix(img,idx):
    #中心座標取得
    h,w = img.shape
    h_center = h / 2 - 1
    w_center = w / 2 - 1
    #sum開始位置を算出
    y1 = int(h_center - idx)
    x1 = int(w_center - idx)
    size = idx * 2 + 2
    y2 = y1 + size
    x2 = x1 + size
    all = img[y1:y2,x1:x2].sum()
    #引き算する合計算出位置
    if(idx == 0):
        diff = 0
        component = all / (size*size)
    else:
        dy1 = int(h_center - (idx-1))
        dx1 = int(w_center - (idx-1))
        dsize = idx * 2
        dy2 = dy1 + dsize
        dx2 = dx1 + dsize
        diff = img[dy1:dy2,dx1:dx2].sum()
        component = (all -diff) / ((size*size) - (dsize*dsize))

    return component



#path設定
input_path = "input\\"
input_name = "noize2.png"
output_path = "output\\"

#画像読み込み
img = cv2.imread(input_path+input_name,0)
print(img.shape)
h,w = img.shape

#fft設定
fftsize = max(h,w)
print(fftsize)

#fft
z = fft.fftshift(fft.fft2(img,(fftsize,fftsize)))
output_name = "fft.png"
#plt.plot(np.abs(z))
plt.plot(np.abs(z))
plt.savefig(output_path + output_name)
'''
#fftの虚数部を実数化
z_abs = np.absolute(z)
#次に対数をとるので1未満は1にする
z_abs[z_abs < 1] = 1
#power spectlum算出
P = np.log10(z_abs)
'''
tmp = np.abs(z)
print("ochi",np.amax(tmp))
tmp2 = (tmp / np.amax(tmp)) * 255
P = tmp2.astype(np.uint8)
img_out = Image.fromarray(P)
output_name = "spectrum_ochi.png"
img_out.save(output_path + output_name)

#power spectrum算出
tmp = np.log(np.abs(z)+1)
tmp2 = tmp / np.amax(tmp)*255
#P = 20 * np.log(np.abs(z)+1)
P = tmp2.astype(np.uint8)
'''
#[0,255]に正規化
P_norm = P / np.amax(P)
y = np.uint8(np.around(P_norm*255))
print(y.shape) 
'''
y = np.uint8(P)
#画像出力
img_out = Image.fromarray(y)
output_name = "spectrum.png"
img_out.save(output_path + output_name)

#------周波数パワーを算出------
#画素値を円周上に取得し1次元配列にする
spectrum_size = y.shape
roop_num = int(spectrum_size[0] / 2)
normalize_power = np.zeros((roop_num))
for i in range(roop_num):
    normalize_power[i] = getOutPix(np.abs(z),i)

print(normalize_power)
plt.figure()
plt.plot(normalize_power[50:])
#plt.show()
output_name = "freq.png"
plt.savefig(output_path + output_name)


#fft.freqによるパワースペクトル算出
time_step = 1
freqs = np.fft.fftfreq(h,time_step)
idx = np.argsort(freqs)
ps = np.abs(np.fft.fft(img))**2
plt.figure()
plt.plot(freqs[idx],ps[idx])
output_name = "freq_pack.png"
plt.savefig(output_path+output_name)

#fft結果のx応答のみを算出する(=y波0)
idx = int(h/2 -1)
tmp = z[idx]
print(tmp)
x_spectrum = np.abs(tmp)
plt.figure()
plt.plot(x_spectrum[:idx+1])
output_name = "xspectrum.png"
plt.savefig(output_path+output_name)