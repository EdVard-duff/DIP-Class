from typing import ForwardRef
import torch
from torch import nn
from torch._C import import_ir_module
import torch.nn.functional as F
import torchvision
import os

from resnet_4ch import resnet
class Config(object):
    pretrained_model_path = './pretrained_models'

    backbone = 'resnet34'

opt = Config()

class SelfConsistNet(nn.Module):
    def __init__(self,backbone_pretrained = True) -> None:
        super(SelfConsistNet,self).__init__()
        ## set backbone
        resnet_layers = int(opt.backbone.split('resnet')[-1]) #34
        backbone = resnet(resnet_layers,
                          backbone_pretrained,
                          os.path.join(opt.pretrained_model_path, opt.backbone+'.pth'))
        # drop pool layer and fc layer, resnet34 layer4 output shape: b,512,8,8
        features = list(backbone.children())[:-2]
        backbone = nn.Sequential(*features)
        self.backbone = backbone

        # source feature map

    def forward(self,img,w,h):
        batch_size = img.shape[0] # 128
        '''
        source feature map
        '''
        source_feature = None
        
        source_feature = self.backbone(img) 
        pass


if __name__ == '__main__':
    pass