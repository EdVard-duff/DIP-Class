from deepfack_dataset import get_train_dataloader, get_test_dataloader
from consistency_net import SelfConsistNet
from config import opt

import os
import time
import datetime
from tensorboardX import SummaryWriter
import torch
from tqdm import tqdm
import shutil
import csv
import numpy as np
from torch import nn
from sklearn.metrics import roc_auc_score, average_precision_score

global test_results, best_auc, best_ap
test_results = []
best_auc = 0
best_ap = 0

def Auc(preds, gts):
    auc = roc_auc_score(gts,preds[:,0])
    ap = average_precision_score(gts,preds[:,0],pos_label=0)
    return auc, ap

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
    for batch_index, (img, mask, label, video_name) in enumerate(train_loader):
        fetchdata_time = time.time() - fetchdata_time
        img, mask, label = img.to(device), mask.to(device), label.to(device)

        forward_time = time.time()

        logits, consis_volumn = net(img)

        forward_time = time.time() - forward_time

        # 真实的一致性矩阵 
        mask = mask.squeeze(1)
        mask = mask.flatten(1,2)
        gts_volumn = torch.zeros(opt.batch_size,256,256)
        for i in range(256):
            gts_volumn[:,:,i] = 1- abs(mask[:,i].unsqueeze(1)-mask)
        classify_loss = criterion_of_class(logits, label) 
        consis_loss = criterion_of_consis(consis_volumn, gts_volumn)

        loss = classify_loss + opt.loss_weight * consis_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        _, pre_label = logits.max(1)
        total += label.size(0)
        correct += pre_label.eq(label).sum().item()
        iteration = epoch * len(train_loader) + batch_index
        if (batch_index + 1) % opt.display_freq == 0:
            cur_lr = optimizer.param_groups[0]['lr']
            avg_acc = correct / total
            avg_loss = train_loss / (batch_index + 1)
            writer.add_scalar('Train/loss', avg_loss, iteration)
            writer.add_scalar('Train/accuracy',  avg_acc,  iteration)
            writer.add_scalar('Train/learning_rate', cur_lr, iteration)

            time_per_batch = (time.time() - start) / (batch_index + 1.)
            last_batches   = (opt.epochs - epoch - 1) * len(train_loader) + (len(train_loader) - batch_index - 1)
            last_time = int(last_batches * time_per_batch)
            time_str  = str(datetime.timedelta(seconds=last_time))

            print(
                "===  step: [{:3}/{}], loss: {:.3f} | acc: {:6.3f} | lr: {:.6f} | estimated last time: {} ===".format(
                    batch_index + 1, len(train_loader), avg_loss, avg_acc, cur_lr, time_str))

        batch_time = time.time() - batch_time
        batch_time = time.time()
        fetchdata_time = time.time()

def test(test_loader, net, criterion_of_class, criterion_of_consis, optimizer, epoch, device, writer):
    global best_auc, best_ap

    net.eval()
    test_loss = 0
    correct = 0

    total = 0
    preds = []
    gts = []
    video_names = []

    m = nn.Softmax(dim=1)

    print("===  Validate [{}/{}] ===".format(epoch + 1, opt.epochs))
    with torch.no_grad():
        for batch_index, (img, label, video_name) in enumerate(tqdm(test_loader)):
            img, label = img.to(device), label.to(device)

            logits, consis_volumn = net(img)
            _, pre_label = logits.max(1)
            logits = m(logits)
            preds.extend(logits.cpu().numpy())
            gts.extend(label.cpu().numpy())
            total += label.size(0)
            video_names.extend(video_name)

            correct += pre_label.eq(label).sum().item()
    
    avg_acc = correct / total
    idx = [0]
    avr_preds = []
    avr_gts = []
    img_num = len(video_names)

    for i in range(1,img_num):
        if video_names[i] != video_names[i-1]:
            idx.append(i)
        else:
            pass   

    idx.append(img_num)

    for i in range(1,len(idx)):
        avr_preds.append(np.mean(preds[idx[i-1]:idx[i]],axis=0))
        avr_gts.append(gts[idx[i-1]])
    
    avr_preds = np.array(avr_preds)
    print(avr_preds)
    auc, ap = Auc(avr_preds,avr_gts)

    print("Test on {} images, local:auc={:.3f},ap={:.3f}".format(total, auc, ap))
    writer.add_scalar('Test/auc', auc, epoch)
    writer.add_scalar('Test/ap', ap, epoch)

    if auc > best_auc:
        best_auc = auc
        checkpoint_path = os.path.join(opt.checkpoint_dir, 'best-auc.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print('Update AUC checkpoint, best_auc={:.3f}'.format(best_auc))

    if ap > best_ap:
        best_ap = ap
        checkpoint_path = os.path.join(opt.checkpoint_dir, 'best-ap.pth')
        torch.save(net.state_dict(), checkpoint_path)
        print('Update AP checkpoint, best_ap={:.3f}'.format(best_ap))

    if epoch % opt.save_freq == 0:
        checkpoint_path = os.path.join(opt.checkpoint_dir, f'model-{epoch}.pth')
        torch.save(net.state_dict(), checkpoint_path)

    writer.add_scalar('Test/best_auc', best_auc, epoch)
    writer.add_scalar('Test/best_ap', best_ap, epoch)
    global test_results
    test_results.append([epoch, auc, ap, avg_acc])


def write_test_results():
    global test_results
    csv_path = os.path.join(opt.exp_path, '..', '{}.csv'.format(opt.exp_name))
    header = ['epoch', 'AUC', 'AP','ACC']
    epoches = list(range(len(test_results)))
    rows = [header] + test_results
    metrics = [[] for i in header]
    for result in test_results:
        for i,r in enumerate(result):
            metrics[i].append(r)
    for name,m in zip(header, metrics):
        if name == 'epoch':
            continue
        index = m.index(max(m))
        title = 'best {}(epoch-{})'.format(name, index)
        row = [l[index] for l in metrics]
        row[0] = title
        rows.append(row)
    with open(csv_path, 'w') as f:
        cw = csv.writer(f)
        cw.writerows(rows)
    print('Save result to ', csv_path)

if __name__ == "__main__":
    device = torch.device("cuda:{}".format(opt.gpu_id))
    opt.create_path()
    for file in ['config.py', 'deepfack_dataset.py', 'consistency_net.py', 'train.py']:
        shutil.copy(file, opt.exp_path)
        print('backup ', file)
    
    net = SelfConsistNet().to(device)
    criterion_for_class = torch.nn.CrossEntropyLoss()
    criterion_for_consis = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(net.parameters(), opt.base_lr) # betas:Default
    # 待找合适的API
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=opt.lr_milestones, gamma=opt.lr_gamma)
    
    test_loader = get_test_dataloader()
    print(("=======  Training  ======="))
    writer = SummaryWriter(log_dir=opt.log_dir)
    for epoch in range(opt.epochs):
        train_loader = get_train_dataloader()
        train(train_loader, net, criterion_for_class, criterion_for_consis, optimizer, epoch, device, writer)
        if epoch == 0 or (epoch + 1) % opt.eval_freq == 0 or epoch == opt.epochs - 1:
            test(test_loader, net, criterion_for_class, criterion_for_consis, optimizer, epoch, device, writer)
            write_test_results()            
        lr_scheduler.step()
    print(( "=======  Training Finished.Best AUC={:.3f}, best AP={:.1%}========".format(best_auc, best_ap)))