# 数据集文件待补充
from XXXdataset import get_train_dataloader, get_test_dataloader
from consistency_net import SelfConsistNet
from config import opt

import os
import time
import datetime
from tensorboardX import SummaryWriter
import torch
import csv
from tqdm import tqdm
import shutil

import torch
import csv
from tqdm import tqdm

def train(train_loader, net, criterion_of_class, criterion_of_consis, optimizer, epoch, device, writer):
    start = time.time()
    net.train()
    train_loss = 0
    correct = 0
    total = 0


    print("\n===  Epoch: [{}/{}]  === ".format(epoch + 1, opt.epochs))
    fetchdata_time = time.time()
    forward_time = time.time()
    batch_time   = time.time()

    for batch_index, (img, label, w , h) in enumerate(train_loader):
        img, w, h = img.to(device), w.to(device), h.to(device)

        forward_time = time.time()

        logits, consis_volumn = net(img,w,h)

        # 真实的一致性矩阵 
        # gts_volumn = 
        classify_loss = criterion_of_class(logits, label) 
        # 可能要展开
        consis_loss = criterion_of_consis(consis_loss, gts_volumn)

        loss = classify_loss + opt.loss_weight * consis_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, pre_label = logits.max(1)
        total += label.size(0)


        batch_time = time.time() - batch_time
        # print('Time cost: fetch batch data:{:.4f}s, network forward:{:.4f}s, batch time:{:.4f}s'.format(
        #     fetchdata_time, forward_time, batch_time
        # ))
        batch_time = time.time()
        fetchdata_time = time.time()

#待写
def test():
    pass 

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(opt.gpu_id))
    ## 备份
    
    net = SelfConsistNet().to(device)
    criterion_for_class = torch.nn.CrossEntropyLoss()
    criterion_for_consis = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(net.parameters(), opt.base_lr) # betas:Default
    # 待找合适的API
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=opt.lr_gamma)

    train_loader = get_train_dataloader()

    print(("=======  Training  ======="))
    writer = SummaryWriter(log_dir=opt.log_dir)
    for epoch in range(opt.epochs):
        train(train_loader, net, criterion_for_class, criterion_for_consis, optimizer, epoch, device, writer)
        '''
        test
        if epoch == 0 or (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
        test_loader = get_test_dataloader()
        test(test_loader, net, criterion, optimizer, epoch, device, writer)
        write_test_results()
        '''

        lr_scheduler.step()
    print(("=======  Training Finished.Best F1={:.3f}, best balanced accuracy={:.1%}========".format(best_f1, best_acc)))
