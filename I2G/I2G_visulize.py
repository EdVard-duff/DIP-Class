from skimage.io.collection import ImageCollection
from blending import blend_src2dst
import numpy as np
import cv2


size = 7
for i in range(size):
    for j in range(size):
        src = f'man{i}'
        dst = f'man{j}'
        img_src, img_dst, _, img_dst_mix_grad, _ = blend_src2dst(src,dst)
        if j==0:
            img_array = np.hstack([img_src,img_dst_mix_grad])
        else:
            img_array = np.hstack([img_array,img_dst_mix_grad])
    if i==0:
        img_all_array = img_array
    else:
        img_all_array = np.vstack([img_all_array,img_array])

i = 0
img_all_dst = np.ones((256,256,3))*255
for j in range(size):
    src = f'man{i}'
    dst = f'man{j}'
    _, img_dst, _, _, _ = blend_src2dst(src,dst) 
    img_all_dst = np.hstack([img_all_dst,img_dst])

img_all_array = np.vstack([img_all_dst,img_all_array])   
cv2.imwrite('I2G/vis.jpg',img_all_array)
#cv2.imshow('I',img_all_array)
#cv2.waitKey(0)