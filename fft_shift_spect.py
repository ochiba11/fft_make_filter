import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as fft
import cv2
import os
from PIL import Image

#画像imgの引数idxの枠組みのみの画素値を合計する
#pixel数で正規化し戻り値とする
def getOutPix(img,idx):
    outpix = 0
    for i in range(idx+1):
        if(i == idx):
            outpix = outpix + img[i,i]
        else:
            outpix = outpix + img[i,idx] + img[idx,i]
    
    outpix = outpix / ((idx+1)*2 -1)
    return outpix

#path設定
input_path = "input\\"
input_name = "noize.png"
output_path = "output\\"
output_name = "fft.png"

#画像読み込み
img = cv2.imread(input_path+input_name,0)
print(img.shape)
h,w = img.shape

#fft設定
fftsize = max(h,w)
print(fftsize)

#fft
z = fft.fftshift(fft.fft2(img,(fftsize,fftsize)))
#fftの虚数部を実数化
z_abs = np.absolute(z)
#次に対数をとるので1未満は1にする
z_abs[z_abs < 1] = 1
#power spectlum算出
P = np.log10(z_abs)
#[0,255]に正規化
P_norm = P / np.amax(P)
y = np.uint8(np.around(P_norm*255))
print(y.shape)
#画像出力
img_out = Image.fromarray(y)
img_out.save(output_path + output_name)

#------周波数パワーを算出------
#まずpower speclumの第4象限を取得
s_x = int(fftsize / 2)
s_y = int(fftsize / 2)
QD4 = y[s_y:fftsize,s_x:fftsize]
#画素値を円周上に取得し1次元配列にする
Incremnet = 2
power_size = fftsize - s_x
normalize_power = np.zeros((power_size))
for i in range(power_size):
    normalize_power[i] = getOutPix(QD4,i)

print(normalize_power)
plt.plot(normalize_power)
plt.show()