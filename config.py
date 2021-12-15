import os
class Config(object):
    video_root = './dataset' 
    origin_train = './dataset/origin_train.csv'
    

    pretrained_model_path = './pretrained_models'

    class_num = 2
    num_workers = 4
    batch_size = 64

    #train
    img_per_frame = 1 # 每个视频选取的图片数目
    base_lr = 5e-5
    #lr_milestones = [10, 16]
    #lr_gamma = 0.1
    epochs = 150
    eval_freq = 1
    save_freq = 5
    display_freq = 10
    betas = (0.9,0.999)

    # baseline
    backbone = 'resnet34'
    loss_weight = 10
    deepfake_method_str = ['deepfakes','face2face','faceswap','neural']
    deepfake_method = 0 # one of [None,0,1,2,3]

    imageNet_normalization = True

    prefix = backbone

    if (deepfake_method != None) and \
        (isinstance(deepfake_method, int)) and \
        (deepfake_method < len(deepfake_method_str)):
        prefix += ('+' + deepfake_method_str[deepfake_method])
        
    exp_root = os.path.join(os.getcwd(), './experiments/ablation_study/')
    exp_name = prefix
    exp_path = os.path.join(exp_root, prefix)
    while os.path.exists(exp_path):
        index = os.path.basename(exp_path).split(prefix)[-1].split('repeat')[-1]
        try:
            index = int(index) + 1
        except:
            index = 1
        exp_name = prefix + ('_repeat{}'.format(index))
        exp_path = os.path.join(exp_root, exp_name)
    print('Experiment name {} \n'.format(os.path.basename(exp_path)))
    checkpoint_dir = os.path.join(exp_path, 'checkpoints')
    log_dir = os.path.join(exp_path, 'logs')

    def create_path(self):
        print('Create experiments directory: ', self.exp_path)
        os.makedirs(self.exp_path)
        os.makedirs(self.checkpoint_dir)
        os.makedirs(self.log_dir)

opt = Config()