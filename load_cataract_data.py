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
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from models import CataractDenseNet
import torchvision as tv
from torch import nn, optim
from torch.autograd import Variable

#%%
class CataractDataset(Dataset):
     
    def __init__(self, csv_path, transform=None):
        self.csv_path = csv_path
        self.raw_csv_dict = {}
        for i in range(25):
            if i < 9:
                csv_file_name = 'train' + '0' + str(i+1)
            else:
                csv_file_name = 'train' + str(i+1)
            self.raw_csv_dict[i+1] = pd.read_csv(csv_path + csv_file_name + '.csv')
        self.transform = transform
        self.img_path = '/media/zfq/本地磁盘/cataract/train/train_img/train_img'
        self.all_img_names = os.listdir(self.img_path)
 
    def __len__(self):
        return 49475
        
    def __getitem__(self, idx):
        img_name = self.all_img_names[idx]
        image = io.imread(self.img_path + '/' + img_name)
        video_idx = img_name.split('_')[0]       #分割出这张图片是第几个视频的
        frame_idx = int(img_name.split('_')[1].split('.')[0]) * 10  #分割出这张图片是第几帧
        labels = self.raw_csv_dict[int(video_idx)].iloc[frame_idx-1, 1:].as_matrix().astype('float')
        sample = {'image': image, 'labels': labels}

        if self.transform:
            sample = self.transform(sample)
    
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        image, labels = sample['image'], sample['labels']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        image = image.transpose((2, 0, 1))
        return {'image': torch.from_numpy(image),
                'labels': torch.from_numpy(labels)}
        


def show_img(sample_batched):
    """Show image with labels for a batch of samples."""
    images_batch = sample_batched['image']
#    grid = utils.make_grid(images_batch)
    img1 = images_batch[0,:,:,:]
    img2 = images_batch[1,:,:,:]
    img3 = images_batch[2,:,:,:]
    img4 = images_batch[3,:,:,:]
    plt.imshow(img1.numpy().transpose((1, 2, 0)))
    plt.imshow(img2.numpy().transpose((1, 2, 0)))
    plt.imshow(img3.numpy().transpose((1, 2, 0)))    
    plt.imshow(img4.numpy().transpose((1, 2, 0)))
    
    
transformed_dataset = CataractDataset(
                      csv_path='/media/zfq/本地磁盘/cataract/train/train_labels/', 
                      transform=transforms.Compose([
                                 ToTensor()
                                ]))

dataloader = DataLoader(transformed_dataset, batch_size=4,
                        shuffle=True, num_workers=4)

#for i_batch, sample_batched in enumerate(dataloader):
#    print(i_batch, sample_batched['image'].size(),
#          sample_batched['labels'].size())
#    if i_batch == 3:
#        break
    
    
growth_rate=12
n_epochs=100
batch_size=4
lr=0.01
wd=0.0001
momentum=0.9

model = CataractDenseNet(
        growth_rate = growth_rate,
        block_config=[6, 24, 20, 16],
        num_classes=21,
    )
print(model)
model_cuda = model.cuda()

optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=wd)
criterion = nn.SoftMarginLoss()

for epoch in range(1, n_epochs + 1):
    
    for i, sample_batched in enumerate(dataloader):
    #    dataiter = iter(dataloader)
    #    sample_batched= dataiter.next()
        images_batch, labels_batch = sample_batched['image'], sample_batched['labels']
        labels_batch[(labels_batch <= 0)] = -1
        #为了方便计算softmarginloss，要求标签y为1或-1,代表正类和负类，详见softmarginloss
        images_batch = images_batch.type(torch.FloatTensor)
        labels_batch = labels_batch.type(torch.FloatTensor)
                     
        model.zero_grad()
        optimizer.zero_grad()
        
        input_var = Variable(images_batch, volatile=False).cuda()
        target_var = Variable(labels_batch,  volatile=False, requires_grad=False).cuda()
        output_var = model_cuda(input_var)
        
        loss = criterion(output_var, target_var)
        loss.backward()
        optimizer.step()
        
        if i % 10 == 0 :
            print('%s: (Epoch %d of %d) [%04d/%04d]   Loss:%.5f'
                  % ('Train',epoch, n_epochs, i + 1, len(dataloader), loss.data[0]))

    if epoch % 10 == 0:
        torch.save(model.state_dict(), 'tmp/cataract_model' + '_' + str(epoch))

            
#%%
#raw_csv_dict = {}
#csv_path = '/media/zfq/本地磁盘/cataract/train/train_labels/'
#for i in range(25):
#    if i < 9:
#        csv_file_name = 'train' + '0' + str(i+1)
#    else:
#        csv_file_name = 'train' + str(i+1)
#    raw_csv_dict[i+1] = pd.read_csv(csv_path + csv_file_name + '.csv')
#img_path = '/media/zfq/本地磁盘/cataract/train/train_img/train_img'
#all_img_names = os.listdir(img_path)
#img_name = all_img_names[1-1]
#image = io.imread(img_path + '/' + img_name)
#video_idx = img_name.split('_')[0]       #分割出这张图片是第几个视频的
#frame_idx = int(img_name.split('_')[1].split('.')[0]) * 10  #分割出这张图片是第几帧
#label = raw_csv_dict[int(video_idx)].iloc[int(frame_idx)-1, 1:].as_matrix().astype('float')
