import os
class Config(object):
    pretrained_model_path = './pretrained_models'

    backbone = 'resnet34'
    class_num = 2

    lr_rate = 5e-5
    betas = (0.9,0.999)

    loss_weight = 10
opt = Config()