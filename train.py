# -*- coding: utf-8 -*-

from __future__ import print_function
from deeplabv3p import DeepLabv3Plus
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
from torch.utils.data import DataLoader
import random
from fcn import VGGNet, FCN32s, FCN16s, FCN8s, FCNs
from readimages import SegmentDataset
# from CamVid_loader import CamVidDataset

from matplotlib import pyplot as plt
import numpy as np
import time
import sys
import os
from tensorboardX import SummaryWriter
# import cv2 as cv
from PIL import Image  


print('删除log1目录...')
def rm(p):
    #判断输入是否为文件
    if os.path.isfile(p):
        os.remove(p)
        print("删除文件：",p)
    else:
        #删除文件的子文件和文件夹
        fs=os.listdir(p)#获取文件夹里的文件
        for temp in fs:
            rm(p+"/"+temp)
        os.rmdir(p)
        print("删除文件夹：",p)

path = './log1'
rm(path)
os.makedirs(path)



n_class    = 9

batch_size = 8
epochs     = 500
lr         = 1e-3
momentum   = 0
w_decay    = 1e-5
step_size  = 10
gamma      = 0.05
configs    = "FCNs-BCEWithLogits_batch{}_epoch{}_RMSprop_scheduler-step{}-gamma{}_lr{}_momentum{}_w_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
configs    = "Rysander_batch{}_epoch{}_step{}gamma{}_lr{}_momentum{}_decay{}".format(batch_size, epochs, step_size, gamma, lr, momentum, w_decay)
print("Configs:", configs)

# if sys.argv[1] == 'CamVid':
#     root_dir   = "CamVid/"
# else
root_dir = '/data3/Rysander/segment/100178_1_1_3_locnet_stem_crop240_whole/'
train_file = os.path.join(root_dir, "train.csv")
test_file = os.path.join(root_dir, "test.csv")

# val_file = os.path.join(root_dir, "train.csv")

num_class = n_class
means     = np.array([103.399, 116.779, 123.68]) / 255. # mean of three channels in the order of BGR
h, w      = 1024, 2048
train_h   = int(h/2)  # 512
train_w   = int(w/2)  # 1024
val_h     = h  # 1024
val_w     = w  # 2048

# create dir for model
model_dir = '/data3/Rysander/segment/100178_1_1_3_locnet_stem_crop240_whole/models/'
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
model_path = os.path.join(model_dir, configs)


# use_gpu = torch.cuda.is_available()
# num_gpu = list(range(torch.cuda.device_count()))

# if sys.argv[1] == 'CamVid':
#     train_data = CamVidDataset(csv_file=train_file, phase='train')
# else:
train_data = SegmentDataset(csv_file=train_file, phase='train')

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
print('Load over, loaded '+str(len(train_loader)))
print('Load over, loaded '+str(len(train_data)))
# if sys.argv[1] == 'CamVid':
#     val_data = CamVidDataset(csv_file=val_file, phase='val', flip_rate=0)
# else:

val_data = SegmentDataset(csv_file=test_file, phase='test', flip_rate=0)
val_loader = DataLoader(val_data, batch_size=1, num_workers=8)

vgg_model = VGGNet(requires_grad=True, remove_fc=True)
# fcn_model = DeepLabv3Plus(channel=3, num_classes=9)
fcn_model = FCNs(pretrained_net=vgg_model, n_class=n_class)

# if use_gpu:
ts   = time.time()
vgg_model = vgg_model.cuda()
fcn_model = fcn_model.cuda()
    # fcn_model = nn.DataParallel(fcn_model, device_ids=num_gpu)
print("Finish cuda loading, time elapsed {}".format(time.time() - ts))

LossWeight = np.array([1,4,5,6,7,8,9,10,13])
LossWeight = torch.from_numpy(LossWeight.copy()).float().cuda()



class mIoULoss(nn.Module):
    # __constants__ = ['weight', 'ignore_index', 'reduction']

    def __init__(self, weight=None, size_average=True, n_classes = 9 ):
        #, size_average=None, ignore_index=-100,
                 #reduce=None, reduction='mean'):
        #super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        # self.ignore_index = ignore_index
        # def __init__(self, weight=None, size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        # self.loss0 = loss0

    def forward(self, inputs, target, batch = 8, loss0 = 100):
        # inputs => N x Classes x H x W
        # target_oneHot => N x Classes x H x W
        
        N = inputs.size()[0]
        oneHot = target.float()
        # one_Hot = target
        # one_Hot = torch.zeros(batch, self.classes).scatter_(1, target, 1)
        # predicted probabilities for each pixel along channel

        # inputs2 = nn.functional.logsigmoid(inputs)#,dim=1)
        # inter2 = -oneHot*inputs2
    
        inputs = nn.functional.softmax(inputs,dim=1)

        inter2 = -nn.functional.log_softmax(inputs,dim=1)
        inter2 = inter2.gather(1,oneHot)


        # Numerator Product
        inter = inputs * oneHot
        # inter2 = oneHot - inter
        # inter2 = -nn.functional.logsigmoid(1-inter2)
        # inter2 = inter2.view(N,self.classes,-1).sum(2)

        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N,self.classes,-1).sum(2)
        oneHot2 = oneHot.view(N,self.classes,-1).sum(2)


        #Denominator 
        union= inputs + oneHot - (inputs*oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N,self.classes,-1).sum(2)
        # inputs2 = nn.functional.logsigmoid(inputs)

        # loss = -union/(union+0.0001)*nn.functional.logsigmoid((inter + 0.0001)/(union+0.0001))

        loss = -union/(union+0.0001)*nn.functional.log_softmax((inter + 0.0001)/(union+0.0001)+0.001,dim = 1)
        loss2 = oneHot2*(inter2+0.0001)/(oneHot2+0.0001)/(oneHot2+0.0001)
        # print(loss2)
        # loss2 = loss2

        # print(loss[0,:])
        loss = loss[:,1:]
        # loss2 = loss2.mean()
        # print(loss.shape)

        ## Return average loss over classes and batch
        return loss.mean()+loss2.mean()#+nn.functions.cross_entropy(input, target, weight=self.weight,
                               # ignore_index=self.ignore_index, reduction=self.reduction)

criterion1 = mIoULoss()# weight = LossWeight)
criterion2 = nn.CrossEntropyLoss(weight = LossWeight)
# criterion2 = mIoULoss()
# criterion = mIoULoss(weight = LossWeight)#,reduction = 'sum')
optimizer = optim.RMSprop(fcn_model.parameters(), lr=lr, momentum=momentum, weight_decay=w_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)  # decay LR by a factor of 0.5 every 30 epochs

# create dir for score
score_dir = os.path.join("scores", configs)
if not os.path.exists(score_dir):
    os.makedirs(score_dir)
IU_scores    = np.zeros((epochs, n_class))
pixel_scores = np.zeros(epochs)


def train():
    i0 = 1
    loss0 = 100
    for epoch in range(epochs):
        scheduler.step()

        ts = time.time()
        for iterr, batch in enumerate(train_loader):
            valuation = batch['v'].cpu().numpy()

            if (np.max(valuation) >= 0) or epoch % 4 == 3:
                # print(iterr,np.max(valuation))
    
                optimizer.zero_grad()
    
                # if use_gpu:
                inputs = Variable(batch['X'].cuda())
                labels = Variable(batch['l'].cuda())
                valuation = Variable(batch['v'].cuda())
    
                outputs = fcn_model(inputs)
    
                loss1 = criterion1(outputs, labels , loss0 = loss0) #, batch = batch_size)
                # loss2 = criterion2(outputs, valuation)
                loss = loss1 #+ loss2
                loss.backward()
                
                optimizer.step()
                # loss0 = data.np(loss1[0])
  
                if iterr % 10 == 0:
                    print("epoch{}, iter{}, loss: {}".format(epoch, iterr, loss.item()))
                    writer.add_scalar('iter/loss',loss.item(),i0)
                    i0 += 1
                    # print(inputs.shape,labels.shape,outputs.shape)
    
                if iterr % 200 == 199:
                    val(epoch)


        print("Finish epoch {}, time elapsed {}".format(epoch, time.time() - ts))
        torch.save(fcn_model, model_path)
        val(epoch)
    writer.close()

        


def val(epoch):
    # fcn_model.eval()  #dropout层失效
    total_ious = []
    pixel_accs = []
    ran = int(100*random.random())
    print('starting valuation ...')
    for iter, batch in enumerate(val_loader, 0):
        # print(iter)
        # if iter % 100 == 0:
            # print(str(iter)+' batches valuated.')
    
        inputs = Variable(batch['X'].cuda())

        output = fcn_model(inputs)
        output = output.data.cpu().numpy()

        N, _, h, w = output.shape
        pred = output.transpose(0, 2, 3, 1).reshape(-1, n_class).argmax(axis=1).reshape(N, h, w)

        target = batch['v'].cpu().numpy().reshape(N, h, w)
        labels = batch['l'].cpu().numpy()
        for p, t in zip(pred, target):
            total_ious.append(iou(p, t))
            pixel_accs.append(pixel_acc(p, t))

        output2 = pred[0,:,:]/9*255
        target2 = target[0,:,:]/9*255
        input2 = inputs.data.cpu().numpy()
        input2 = input2[0,0,:,:]
        
        input2 = np.stack((input2,input2,input2),2)
        if np.max(output2)*np.max(target2) != 0:
            if iter % 100 == ran:
                if epoch >= 0:
                    print(np.max(output2),np.max(target2),np.max(input2))
                    im = Image.fromarray(np.uint8(output2))
                    im.save("/data3/Rysander/segment/testimg/e{}i{}a.jpg".format(epoch,iter))
                    im2 = Image.fromarray(np.uint8(target2))
                    im2.save("/data3/Rysander/segment/testimg/e{}i{}b.jpg".format(epoch,iter))
                    im3 = Image.fromarray(np.uint8(input2),mode = 'RGB')
                    im3.save("/data3/Rysander/segment/testimg/e{}i{}c.jpg".format(epoch,iter))
                    print('image saved')
    print('valuation over')
    

    # Calculate average IoU
    total_ious = np.array(total_ious).T  # n_class * val_len
    ious = np.nanmean(total_ious, axis=1)
    pixel_accs = np.array(pixel_accs).mean()
    print("epoch: {}, pix_acc: {}, meanIoU: {}, IoUs: {}".format(epoch, pixel_accs, np.nanmean(ious), ious))
    writer.add_scalar('epoch/meaniou',np.nanmean(ious),epoch)
    writer.add_scalars('epoch/ious', {'0':ious[0],
                                    '1':ious[1],
                                    '2':ious[2],
                                    '3':ious[3],
                                    '4':ious[4],
                                    '5':ious[5],
                                    '6':ious[6],
                                    '7':ious[7],
                                    '8':ious[8]}, epoch)

    IU_scores[epoch] = ious
    np.save(os.path.join(score_dir, "meanIU"), IU_scores)
    pixel_scores[epoch] = pixel_accs
    np.save(os.path.join(score_dir, "meanPixel"), pixel_scores)


# borrow functions and modify it from https://github.com/Kaixhin/FCN-semantic-segmentation/blob/master/main.py
# Calculates class intersections over unions
def iou(pred, target):
    ious = []
    for cls in range(n_class):
        pred_inds = pred == cls
        target_inds = target == cls
        intersection = pred_inds[target_inds].sum()
        union = pred_inds.sum() + target_inds.sum() - intersection
        if union == 0:
            ious.append(float('nan'))  # if there is no ground truth, do not include in evaluation
        else:
            ious.append(float(intersection) / max(union, 1))
        # print("cls", cls, pred_inds.sum(), target_inds.sum(), intersection, float(intersection) / max(union, 1))
    return ious


def pixel_acc(pred, target):
    correct = (pred == target).sum()
    total   = (target == target).sum()
    return correct / total











if __name__ == "__main__":
    # val(0)  # show the accuracy before training
    writer = SummaryWriter('log1')
    train()
    writer.close()