from typing import ForwardRef
import torch
import numpy as np
from torch import nn
from torch._C import import_ir_module
import torch.nn.functional as F
import torchvision
import os

from config import opt

# 原始数据的数据增强
from resnet_4ch import resnet

def pairwise_similarity(x): # b,c',16,16
    y = torch.zeros(size=(16,16,opt.batch_size,16,16)) # 16,16,b,16,16
    for i in range(16):
        for j in range(16):
            y[i,j,:,:,:] = torch.sum(x[:,:,i,j,None,None] * x,dim=1)/np.sqrt(opt.embed_size)
    y = torch.sigmoid(y).permute(2,0,1,3,4)
    return y

class SelfConsistNet(nn.Module):
    def __init__(self,backbone_pretrained = True) -> None:
        super(SelfConsistNet,self).__init__()
        ## set backbone
        resnet_layers = int(opt.backbone.split('resnet')[-1]) #34
        backbone = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))
        # drop pool layer and fc layer, resnet34 layer4 output shape: b,512,8,8
        features = list(backbone.children())
        #feature[6]: CONV4; feature[7]: CONV5
        backbone = nn.Sequential(*features[:7]) # b,256,16,16 
        self.backbone = backbone
       
        # source feature map
        self.feature_size = 256
        '''
        consistency
        '''
        self.embedding_layer = nn.Conv2d(256,opt.embed_size,kernel_size=(1,1))

        '''
        Class prediction
        '''
        self.resblock = nn.Sequential(features[7]) # b,512,8,8 ,可能和原文不一样
        self.avgpool1x1 = nn.AdaptiveAvgPool2d(1) 
        self.prediction_head = nn.Linear(512,opt.class_num,bias=False)

        

    def forward(self,img):
        # img:3,256,256
        batch_size = img.shape[0] # 128
        '''
        source feature map
        '''
        source_feature = None
        
        source_feature = self.backbone(img) # b,256,16,16
        '''
        class predict
        '''
        predict_feature = self.resblock(source_feature) # b,512,8,8
        predict_feature = self.avgpool1x1(predict_feature) # b,512,1,1
        predict_feature = predict_feature.flatten(1) #b,512

        prediction = self.prediction_head(predict_feature) # 2,yes or no

        '''
        cal consistency
        '''
        embed_feature = self.embedding_layer(source_feature) #b,opt.embed_size,16,16
        embed_feature = embed_feature.flatten(2,3)
        
        consistency_volume = torch.zeros(batch_size,256,256) # b, 256, 256
        for i in range(256):
            consistency_volume[:,:,i] = torch.sum(embed_feature[:,:,i].unsqueeze_(2) * embed_feature,dim=1) / np.sqrt(opt.embed_size)
        consistency_volume = torch.sigmoid(consistency_volume)

        return prediction ,consistency_volume

if __name__ == '__main__':
    device = torch.device('cuda:0')
    b = 4
    img = torch.randn(b,3,256,256).to(device)
    w = h = (torch.ones(b) * 256).to(device)
    model = SelfConsistNet(backbone_pretrained=False).to(device)
    tmp, local_pre = model(img)
    print(local_pre.shape)
    print(tmp.shape)