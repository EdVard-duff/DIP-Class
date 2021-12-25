import csv
import os
from tqdm import tqdm

import numpy as np
from PIL import Image
from scipy.signal import convolve2d
from torch import Size
from torchvision import transforms
from skimage.metrics import structural_similarity
import time
import cv2

preprocess = transforms.Compose([
    transforms.Resize((256,256)),
    #transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def matlab_style_gauss2D(shape=(3, 3), sigma=0.5):
    """
    2D gaussian mask - should give the same result as MATLAB's
    fspecial('gaussian',[shape],[sigma])
    """
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]
    h = np.exp(-(x * x + y * y) / (2. * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    sumh = h.sum()
    if sumh != 0:
        h /= sumh
    return h


def filter2(x, kernel, mode='same'):
    return convolve2d(x, np.rot90(kernel, 2), mode=mode)


def compute_ssim(im1, im2, k1=0.01, k2=0.03, win_size=11, L=255):
    if not im1.shape == im2.shape:
        raise ValueError("Input Imagees must have the same dimensions")
    if len(im1.shape) > 2:
        raise ValueError("Please input the images with 1 channel")

    M, N = im1.shape
    C1 = (k1 * L) ** 2
    C2 = (k2 * L) ** 2
    window = matlab_style_gauss2D(shape=(win_size, win_size), sigma=1.5)
    window = window / np.sum(np.sum(window))

    if im1.dtype == np.uint8:
        im1 = np.double(im1)
    if im2.dtype == np.uint8:
        im2 = np.double(im2)

    mu1 = filter2(im1, window, 'valid')
    mu2 = filter2(im2, window, 'valid')
    mu1_sq = mu1 * mu1
    mu2_sq = mu2 * mu2
    mu1_mu2 = mu1 * mu2
    sigma1_sq = filter2(im1 * im1, window, 'valid') - mu1_sq
    sigma2_sq = filter2(im2 * im2, window, 'valid') - mu2_sq
    sigmal2 = filter2(im1 * im2, window, 'valid') - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigmal2 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return np.mean(np.mean(ssim_map))


if __name__ == "__main__":
    deepfake_method = 'deepfakes'
    data_root = './dataset/trainset'
    dirs1 = os.listdir(os.path.join(data_root,deepfake_method))
    dirs2 = os.listdir('./dataset/trainset/original')
    for videoidx in tqdm(range(len(dirs1))):
        path1 = os.path.join(data_root,deepfake_method,dirs1[videoidx])
        path2 = os.path.join(data_root,'original',dirs2[videoidx])
        images = os.listdir(path1)
        images2 = os.listdir(path2)
        maskfloder = os.path.join(data_root,'mask',deepfake_method,dirs1[videoidx])
        if not os.path.exists(maskfloder):
            os.mkdir(maskfloder)
        data = [path1,maskfloder,1]
        #with open("train.csv", "a+", newline='') as f:
        #    csv_write = csv.writer(f)
        #    csv_write.writerow(data)
        for image in images:
            if "npy" not in image:
                if image in images2:
                    try:
                        maskpath = os.path.join(maskfloder,image+'.npy')           
                        if not os.path.exists(maskpath):
                            im1 = Image.open(os.path.join(path1,image)) 
                            im2 = Image.open(os.path.join(path2,image))
                            im1 = preprocess(im1)
                            im2 = preprocess(im2)
                            im1 = np.asanyarray(im1)
                            im2 = np.asanyarray(im2)
                            im1 = im1.transpose(1,2,0)
                            im2 = im2.transpose(1,2,0)
                            im1 = cv2.cvtColor(im1, cv2.COLOR_RGB2BGR)
                            im2 = cv2.cvtColor(im2, cv2.COLOR_RGB2BGR)
                            grayA = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
                            grayB = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
                            # 计算两个灰度图像之间的结构相似度
                            (score, diff) = structural_similarity(grayA, grayB, win_size=101, full=True)
                            #diff = (diff * 255).astype("uint8")
                            #print(diff)
                            #print("SSIM:{}".format(score))
                            sub = cv2.resize(diff, (16,16), cv2.INTER_LINEAR)
                            np.save(maskpath, sub)
                        else:
                            print("skip this one")
                            continue
                    except Exception:
                        continue
                elif image not in images2:
                     os.remove(os.path.join(path1,image)) 
            else:
                os.remove(os.path.join(path1,image))