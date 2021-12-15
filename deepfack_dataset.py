import csv
import math
import os
import shutil
import random

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.functional import Tensor
from torch.utils.data import DataLoader, Dataset

from config import opt
torch.random.manual_seed(1)

class ImageDataset(Dataset):
    def __init__(self,istrain=True):
        self.istrain = istrain

        self.video_path = []
        self.video_type = []
        #添加真实图片
        with open(opt.origin_train) as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]
        for row in reader:
            self.video_path.append(row[0])
            self.video_type.append(0)

        #添加伪造方法
        if opt.deepfake_method is None:
            pass
        else:
            with open(os.path.join(opt.video_root,
                    opt.deepfake_method_str[opt.deepfake_method]+'.csv')) as f:
                reader = csv.reader(f)
                reader = list(reader)
                reader = reader[1:]
            for row in reader:
                self.video_path.append(row[0])
                self.video_type.append(1)
       
        self.labels = []
        self.images_path = []
        self.img_type = [] # real, deepfake and I2G
        self.mask_path = [] #16*16
        self.video = []
        for i in range(len(self.video_path)):
            img_list = os.listdir(self.video_path[i])
            random.shuffle(img_list)
            for img_name in img_list[:opt.img_per_frame]:
                prefix = img_name.split('.jpg')[0]   
                self.images_path.append(os.path.join(self.video_path[i],img_name))
                self.video.append(self.video_path[i])
                if self.video_type[i] == 0:
                    self.labels.append(0)
                    self.img_type.append(0)
                    self.mask_path.append('None')
                else:
                    self.labels.append(1)
                    self.img_type.append(1)
                    self.mask_path.append(os.path.join(self.video_path[i],
                                        prefix+'.npy'))

        # 这部分还可以添加其它的数据增强的组件
        if opt.imageNet_normalization:
            self.img_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # 归一化到0-1之间了
                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225], inplace=True) 
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ])
        
    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')
        img = self.img_transform(img)

        if self.mask_path[index] == 'None':
            mask = np.ones((16,16))
        else:
            mask = np.load(self.mask_path[index])  #16,16

        label = self.labels[index]
        video_name = self.video[index]
        # I2G待添加
        return img, mask, label, video_name

    def __len__(self):
        return len(self.labels)


class ImageDataset_for_test(Dataset):
    def __init__(self,istrain=False):
        self.istrain = istrain

        self.video_path = []
        self.video_type = []

        #添加伪造方法
        if opt.deepfake_method is None:
            pass
        else:
            with open(os.path.join(opt.video_root,
                        opt.deepfake_method_str[opt.deepfake_method]+'_test.csv')) as f:
                reader = csv.reader(f)
                reader = list(reader)
                reader = reader[1:]
            for row in reader:
                self.video_path.append(row[0])
                self.video_type.append(1)
       
        self.labels = []
        self.images_path = []
        self.img_type = [] # real, deepfake and I2G
        self.video = []
        for i in range(len(self.video_path)):
            img_list = os.listdir(self.video_path[i])
            for img_name in img_list:
                self.images_path.append(os.path.join(self.video_path[i],img_name))
                self.video.append(self.video_path[i])
                self.labels.append(1) # 假设测试集全是假的图片了，暂时不考虑origin_test.csv


        # 这部分还可以添加其它的数据增强的组件
        if opt.imageNet_normalization:
            self.img_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225], inplace=True) 
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ])
        
    def __getitem__(self, index):
        img = Image.open(self.images_path[index]).convert('RGB')
        img = self.img_transform(img)

        label = self.labels[index]
        video_name = self.video[index]
        # I2G待添加
        return img, label, video_name

    def __len__(self):
        return len(self.labels)

def get_train_dataloader():
    trainset = ImageDataset(istrain=True)
    print('Training images', len(trainset))
    train_loader = torch.utils.data.DataLoader(trainset, batch_size=opt.batch_size,
                                               shuffle=True, num_workers=opt.num_workers,
                                               drop_last=True, pin_memory=True)
    return train_loader

def get_test_dataloader():
    testset = ImageDataset_for_test(istrain=False)
    print('Testing images', len(testset))
    test_loader = torch.utils.data.DataLoader(testset, batch_size=opt.batch_size * 2,
                                              shuffle=False, num_workers=opt.num_workers,
                                              drop_last=False, pin_memory=True)
    return test_loader

if __name__ == '__main__':
    train_set = ImageDataset()

    img, mask, label, video_name = train_set.__getitem__(0)
    print(img)
    #print(train_set.mask_path)