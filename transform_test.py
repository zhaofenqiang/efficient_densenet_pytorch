#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 16:08:26 2017

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

#%%     
        
mean = [0.61372, 0.34675, 0.14977]
stdv = [0.18669, 0.17454, 0.11950]
train_transforms = tv.transforms.Compose([
   tv.transforms.ToTensor(),
])   


raw_csv_dict = {}
csv_path='/media/zfq/本地磁盘/cataract/train/train_labels/'
for i in range(25):
    if i < 9:
        csv_file_name = 'train' + '0' + str(i+1)
    else:
        csv_file_name = 'train' + str(i+1)
    raw_csv_dict[i+1] = pd.read_csv(csv_path + csv_file_name + '.csv')
img_path = '/media/zfq/本地磁盘/cataract/train/train_img'
all_img_names = os.listdir(img_path)
img_name = all_img_names[1000]
image = io.imread(img_path + '/' + img_name)
img = Image.open(img_path + '/' + img_name)
video_idx = int(img_name.split('_')[0])       #分割出这张图片是第几个视频的
frame_idx = int(img_name.split('_')[1].split('.')[0])  #分割出这张图片是第几帧
labels = raw_csv_dict[video_idx].iloc[frame_idx-1, 1:].as_matrix().astype('float')

imgt = train_transforms(img)
