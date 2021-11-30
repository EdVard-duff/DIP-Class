import os
class Config(object):
    pretrained_model_path = './pretrained_models'

    backbone = 'resnet34'
    class_num = 2
    without_mask = True

opt = Config()