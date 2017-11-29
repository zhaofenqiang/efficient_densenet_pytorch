#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 17:05:12 2017

@author: zfq
"""

from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from models import CataractDenseNet
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable
from PIL import Image
from sklearn.metrics import roc_curve, auc
from math import isnan

#%%
growth_rate=24
block_config=[6, 12, 32, 32]
n_epochs=20
batch_size=8
base_lr=0.01
wd=0.0001
momentum=0.9
num_init_features=64
tool_weight = [10,2,2,2,2,2,2,4,4,4,5,1,1,2,3,6,6,1,6,6,8]

class CataractDataset(Dataset):
     
    def __init__(self, csv_path, transform=None, labelTransform=None):
        self.csv_path = csv_path
        self.raw_csv_dict = {}
        for i in range(25):
            if i < 9:
                csv_file_name = 'train' + '0' + str(i+1)
            else:
                csv_file_name = 'train' + str(i+1)
            self.raw_csv_dict[i+1] = pd.read_csv(csv_path + csv_file_name + '.csv')
        self.img_path = '/media/xnat/软件/cataract/train/train_img'
        self.all_img_names = os.listdir(self.img_path)
        self.transform = transform
        self.labelTransform = labelTransform
 
    def __len__(self):
        return len(self.all_img_names)
        
    def __getitem__(self, idx):
        img_name = self.all_img_names[idx]
        image = Image.open(self.img_path + '/' + img_name)
        video_idx = int(img_name.split('_')[0])       #分割出这张图片是第几个视频的
        frame_idx = int(img_name.split('_')[1].split('.')[0])  #分割出这张图片是第几帧
        labels = self.raw_csv_dict[video_idx].iloc[frame_idx-1, 1:].as_matrix().astype('float')
       
        if self.transform:
            image = self.transform(image)
        if self.labelTransform:
            labels = self.labelTransform(labels)
        return image, labels
     
        
mean = [0.61372, 0.34675, 0.14977]
stdv = [0.18669, 0.17454, 0.11950]
train_transforms = tv.transforms.Compose([
    tv.transforms.RandomCrop([540, 960], padding=20),
    tv.transforms.RandomHorizontalFlip(),
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=mean, std=stdv),
])   
    
class labelToTensor(object):
    def __call__(self, labels):
        return torch.from_numpy(labels).type(torch.FloatTensor)

train_dataset = CataractDataset(
                      csv_path='/media/xnat/软件/cataract/train/train_labels/', 
                      transform=train_transforms, 
                      labelTransform=labelToTensor())

train_dataloader = DataLoader(train_dataset, batch_size=batch_size,
                        shuffle=True, num_workers=8)

#dataiter = iter(train_dataloader)
#img, labels = dataiter.next()

#for i_batch, sample_batched in enumerate(dataloader):
#    print(i_batch, sample_batched['image'].size(),
#          sample_batched['labels'].size())
#    if i_batch == 3:
#        break
    
model = CataractDenseNet(
        growth_rate = growth_rate,
        block_config = block_config,
        num_classes = 21,
        num_init_features = num_init_features,
    )

#model = tv.models.resnet50(pretrained=True)
#model.fc = nn.Linear(model.fc.in_features, 21)

print(model)
optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=momentum, weight_decay=wd)
criterion = nn.SoftMarginLoss()

model_cuda = model.cuda()
model_cuda.load_state_dict(torch.load('/home/xnat/pytorch_densenet_cataract/tmp/cataract_densenet169_3'))
model_cuda.train()

for epoch in range(1, n_epochs + 1):
  
    if float(epoch) / n_epochs > 0.3:
        lr = base_lr * 0.01
    elif float(epoch) / n_epochs > 0.1:
        lr = base_lr * 0.1
    else:
        lr = base_lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        print(param_group['lr'])

    for i, (img, labels) in enumerate(train_dataloader):
       
        labels[(labels <= 0)] = -1
        #为了方便计算softmarginloss，要求标签y为1或-1,代表正类和负类，详见softmarginloss
        
        model_cuda.zero_grad()
        optimizer.zero_grad()
        
        input_var = Variable(img, volatile=False).cuda()
        target_var = Variable(labels,  volatile=False, requires_grad=False).cuda()
        output_var = model_cuda(input_var)
        
        weight = 1
        for k in range(21):
            weight += tool_weight[k] * torch.sum(labels[:,k] == 1) 
            
        loss = weight * criterion(output_var, target_var)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0 :
            print('%s: (Epoch %d of %d) [%04d/%04d]   Loss:%.5f'
                  % ('Train',epoch, n_epochs, i, len(train_dataloader), loss.data[0]))
        if i % 100 == 0:
            score = 0
            count = 0
            for j in range(output_var.size()[1]):
                predictions = output_var[:,j].cpu().data.numpy()
                truth = target_var[:,j].cpu().data.numpy()
                truth[(truth < 0)] = 0
                truth[(truth > 0)] = 1
                fpr, tpr, _ = roc_curve(truth, predictions, pos_label=1)
                score1 = auc(fpr, tpr)
                if isnan(score1):
                    score += 0
                else:
                    score += score1
                    count += 1
            if count == 0:
                count =1
            print('%s: (Epoch %d of %d) [%04d/%04d]   score:%.5f'
                 % ('Train',epoch, n_epochs, i, len(train_dataloader), score/count))

    torch.save(model.state_dict(), 'tmp/cataract_densenet169' + '_' + str(epoch+1))
    

            
#%%
#raw_csv_dict = {}
#csv_path='/media/xnat/软件/cataract/train/train_labels/'
#for i in range(25):
#    if i < 9:
#        csv_file_name = 'train' + '0' + str(i+1)
#    else:
#        csv_file_name = 'train' + str(i+1)
#    raw_csv_dict[i+1] = pd.read_csv(csv_path + csv_file_name + '.csv')
#img_path = '/media/xnat/软件/cataract/train/train_img'
#all_img_names = os.listdir(img_path)
#img_name = all_img_names[1-1]
#image = io.imread(img_path + '/' + img_name)
#img = Image.open(img_path + '/' + img_name)
#video_idx = int(img_name.split('_')[0])       #分割出这张图片是第几个视频的
#frame_idx = int(img_name.split('_')[1].split('.')[0])  #分割出这张图片是第几帧
#labels = raw_csv_dict[video_idx].iloc[frame_idx-1, 1:].as_matrix().astype('float')


#
#def show_img(img):
#    """Show image with labels for a batch of samples."""
#    images_batch = img
##    grid = utils.make_grid(images_batch)
#    img1 = images_batch[0,:,:,:]
#    img2 = images_batch[1,:,:,:]
#    img3 = images_batch[2,:,:,:]
#    img4 = images_batch[3,:,:,:]
#    plt.imshow(img1.numpy().transpose((1, 2, 0)))
#    plt.imshow(img2.numpy().transpose((1, 2, 0)))
#    plt.imshow(img3.numpy().transpose((1, 2, 0)))    
#    plt.imshow(img4.numpy().transpose((1, 2, 0)))