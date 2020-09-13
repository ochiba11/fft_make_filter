import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import cv2
import os
from PIL import Image

output_path = "output\\"
input_path = "input\\"

#1dspect計算
def dim1_spect_func(abs_z):
    print("---------dim1_spect()---------")
    h,w = abs_z.shape
    size = min(h,w)
    center = size // 2
    len = size - center
    dim1_spect = np.zeros(len)

    for i in range(len):
        print(" i:",i)
        if(i==0):
            dim1_spect[i] = abs_z[center,center]
        else:
            sum = 0
            for j in range(h):
                for k in range(w):
                    r = (j-center)*(j-center) + (k-center)*(k-center)
                    i2 = i * i
                    next_i2 = ((i+1)*(i+1))
                    if((i2 <= r) and (r < next_i2)):
                        sum = sum + abs_z[j,k]
            dim1_spect[i] = sum

    nikist = 300
    lpi = nikist / len
    plt.figure()
    plt.style.use('dark_background')
    plt.xlabel("LineScreenFreq(lpi)")
    plt.ylabel("Power")
    plt.xticks([0,75//lpi,150//lpi,225//lpi,300//lpi],[0,75,150,225,300])
    plt.plot(dim1_spect)
    output_name = "1dim_spect.png"
    plt.savefig(output_path+output_name)
    return dim1_spect

#FFT
def fft_func(img):
    print("---------fft()---------")
    dc = np.average(img)
    #dc = 0
    ac_img = img - dc
    z = np.fft.fftshift(np.fft.fft2(ac_img))

    abs_z = np.abs(z)
    output_name = "fft.png"
    plt.plot(abs_z)
    plt.savefig(output_path + output_name)

    dim2_spect = abs_z
    dim2_spect[dim2_spect < 1] = 1
    dim2_spect = 20*np.log(abs_z)
    img_out = np.uint8(dim2_spect)
    img_out = Image.fromarray(img_out)
    output_name = "2dspect.png"
    img_out.save(output_path + output_name)
    
    return abs_z,dc,dim2_spect,z

#逆fft
def ifft_func(z,dc):
    print("---------ifft()---------")
    ac_img = np.fft.ifft2(np.fft.ifftshift(z))
    img = np.abs(ac_img+dc)
    img_out = Image.fromarray(np.uint8(img))
    output_name = "ifft.png"
    img_out.save(output_path + output_name)
    return img

if __name__ == "__main__":
    #画像読み込み
    input_name = "noize.png" 
    img = cv2.imread(input_path+input_name,0)

    #fft
    abs_z,dc,dim2_spect,z = fft_func(img)

    #1dfft
    dim1_spect = dim1_spect_func(abs_z)

    #ifft
    ret_img = ifft_func(z,dc)