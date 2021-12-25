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
import elasticdeform

from config import opt
from i2g import d_model, blend_src2dst, cal_landmark

torch.random.manual_seed(1)

class ImageDataset(Dataset):
    def __init__(self,istrain=True):
        self.istrain = istrain

        self.video_path = []
        self.video_type = []
        self.npy_path = []
        #添加真实图片
        with open(opt.origin_train) as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]
        for row in reader:
            # origin_train 
            # 第一列video_path,形式 dataset\trainset\original\XXX.mp4
            # 第二列npy路径,形式 dataset\trainset\mask\original\XXX.mp4, original没有mask可以写 None
            self.video_path.append(row[0]) 
            self.npy_path.append(row[1]) 
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
                self.video_path.append(row[0]) # deepfake_train 第一列video_path 
                self.npy_path.append(row[1]) 
                self.video_type.append(1)
       
        self.labels = []
        self.images_path = [] # 这个路径应该可以直接打开图片
        self.img_type = [] # real, deepfake and I2G
        self.mask_path = [] #16*16
        self.video = []
        for i in range(len(self.video_path)):
            img_list = os.listdir(self.video_path[i])
            random.shuffle(img_list)
            for img_name in img_list[:min(len(img_list),opt.img_per_frame)]: #这里可能没有32张
                #prefix = img_name.split('.jpg')[0]   
                self.images_path.append(os.path.join(self.video_path[i],img_name))
                self.video.append(self.video_path[i])
                if self.video_type[i] == 0:
                    self.labels.append(0)
                    self.img_type.append(0)
                    self.mask_path.append('None') # 来自真视频的图片mask路径'None'
                else:
                    self.labels.append(1)
                    self.img_type.append(1)
                    self.mask_path.append(os.path.join(self.npy_path[i], #此处一定要保证每个feepfake的图片都有同名的npy文件
                                        img_name+'.npy'))

        # 这部分还可以添加其它的数据增强的组件
        if opt.imageNet_normalization:
            self.img_transform = transforms.Compose([
                transforms.Resize((256, 256)),
                transforms.ToTensor(), # 归一化到0-1之间了
                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225], inplace=True) 
            ])
            self.i2g_normalization = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406], [0.229, 0.224, 0.225], inplace=True), 
            ])
        else:
            self.img_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
            ])
            self.i2g_normalization = transforms.Compose([
                transforms.ToTensor(),
            ])

        self.mask_trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((16,16))
        ])

        
    def __getitem__(self, index):
        img_type = self.labels[index]
        is_i2g = False
        if img_type==0 and np.random.uniform() < 0.5:
            img_src = cv2.imread(self.images_path[index])
            img_src = cv2.resize(img_src,(256,256),interpolation= cv2.INTER_AREA)
            points_src = cal_landmark(d_model,img_src)
            if len(points_src) == 0: #没得到landmark直接跳过
                pass
            else:
                tar_idx = index
                points_dst = []
                count = 0
                # 还可以添加欧式距离的指标
                while(self.video[tar_idx] == self.video[index]  # 同一个视频
                            or self.labels[tar_idx] == 1 # 或者选到了假的视频
                            or len(points_dst)==0 ):    # 或者没算出landmark
                    tar_idx = np.random.randint(self.__len__())
                    img_dst = cv2.imread(self.images_path[tar_idx])
                    img_dst= cv2.resize(img_dst,(256,256),interpolation= cv2.INTER_AREA)
                    points_dst = cal_landmark(d_model,img_dst)
                    count = count + 1
                    if count == 5: #防止一直找不到合适的
                        break
                if count <= 5:
                    is_i2g = True
    

        if is_i2g:
            try:
                img, mask = blend_src2dst(img_src,img_dst,points_src,points_dst)
                img = self.i2g_normalization(img)
                label = 1
                
                mask = elasticdeform.deform_random_grid(mask,sigma=6,points=4)
                mask = self.mask_trans(mask)
                mask = mask.squeeze(0)
            except:
                img = Image.open(self.images_path[index]).convert('RGB')
                img = self.img_transform(img)
                if self.mask_path[index] == 'None':
                    mask = torch.ones((16,16))
                else:
                    try:
                        mask = torch.from_numpy(np.load(self.mask_path[index],allow_pickle=True))
                    except:
                        mask = torch.rand((1,16,16))

                label = self.labels[index]
        else:        
            img = Image.open(self.images_path[index]).convert('RGB')
            img = self.img_transform(img)

            if self.mask_path[index] == 'None':
                mask = torch.ones((16,16))
            else:
                try:
                    mask = torch.from_numpy(np.load(self.mask_path[index],allow_pickle=True))
                except:
                    mask = torch.ones((16,16))
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
        
        with open(opt.origin_test) as f:
            reader = csv.reader(f)
            reader = list(reader)
            reader = reader[1:]
        for row in reader:
            self.video_path.append(row[0]) # test_video只需要一列
            self.video_type.append(0)

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
        shuffle_idx = list(range(len(self.video_path)))
        random.shuffle(shuffle_idx)

        self.labels = []
        self.images_path = []
        self.img_type = [] # real, deepfake and I2G
        self.video = []
        for i in range(100):
            img_list = os.listdir(self.video_path[shuffle_idx[i]])
            for img_name in img_list:
                self.images_path.append(os.path.join(self.video_path[shuffle_idx[i]],img_name))
                self.video.append(self.video_path[shuffle_idx[i]])
                if self.video_type[shuffle_idx[i]] == 0:
                    self.labels.append(0)
                    self.img_type.append(0)
                else:
                    self.labels.append(1)
                    self.img_type.append(1)

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
    '''
    train_set = ImageDataset()

    img, mask, label, video_name = train_set.__getitem__(0)
    #print(img)
    #print(train_set.mask_path)
    test_set = ImageDataset_for_test()
    img, label, video_name = test_set.__getitem__(0)
    print(video_name)

 
    '''
    train_loader = get_train_dataloader()
    for batch_index, (img, mask, label, video_name) in enumerate(train_loader):
        print(batch_index)
        print(img.shape) 

    '''
    testloader = get_test_dataloader()
    for batch_index, (img, label, video_name) in enumerate(testloader):
        if batch_index % 100 ==0:
            print(img.shape)
    '''


