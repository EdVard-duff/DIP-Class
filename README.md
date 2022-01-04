# 数字图像处理课程 24组

本项目组成员有刘振宸，别风翔。由于数据集过大，链接如下, 同时项目上传在了[github](https://github.com/EdVard-duff/DIP-Class).

## 说明
由于项目本身只是论文复刻，**没有相关的项目演示以及可执行文件**，代码的使用流程在之后具体说明。

## 简介
本项目主要为论文《Learning Self-Consistency for Deepfake Detection》的复现结果，论文见[链接](https://arxiv.org/abs/2012.09311)。
## 代码文件构成
+ 将数据集下载后解压到 ```dataset/``` 目录下
+ ```dlib ``` 和 ```face-alignment``` 存储了人脸 landmark 检测相关代码，```I2G``` 中存储了人脸混合的相关代码，项目中通过根目录下的 ```i2g.py``` 调用
+ ```pretrained_models``` 中为 Resnet 的预训练模型，```resnet-4ch.py``` 储存了 Resent 的网络结构
+ ```faceswap.py, DSSMI.py, download—Faceforensics.py, deepfake_dataset.py``` 为数据集处理的相关代码，可以根据需要自行调整
## 代码使用
+ ```config.py``` 记录了网络的一些配置参数，诸如 embedding 的维数，学习率，显卡 id, 路径等
+ ```deepfake_dataset.py``` 主要用于读取数据集，返回 train_loader 和 test_loader
+ ```consistency_net.py``` 为我们使用的网络
+ ```train.py``` 训练代码

使用时修改 config.py 里诸如显卡id等参数，调用 ```python train.py``` 即可开始训练。


