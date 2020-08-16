import numpy as np
import matplotlib.pyplot as plt
import scipy.fftpack as sfft
import cv2
import os

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
z = sfft.fftshift(sfft.fft2(img,(fftsize,fftsize)))
plt.plot(np.abs(z))
plt.show()
print("z.shape",z.shape)

#ノイズ除去
A = np.ones((fftsize,fftsize))
A[350:,:]=0
A[:250,:]=0
A[:,:250]=0
A[:,350:]=0
plt.plot(np.abs(z*A))
plt.show()
img2 = np.uint8(np.abs(sfft.ifft2(sfft.fftshift(z*A))))
plt.imshow(img2[:h,:w],cmap="gray")
plt.show()

#ノイズ除去した画像を再度fft
img3 = img2[:h,:w]
cv2.imwrite(output_path+output_name,img3)
z2 = sfft.fftshift(sfft.fft2(img3,(fftsize,fftsize)))
plt.plot(np.abs(z2))
plt.show()
print(z.shape)