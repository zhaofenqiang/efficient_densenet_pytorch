#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:41:03 2017
@author: zfq
"""

import torch

import torch.nn as nn
import torch.optim as optim

import torch.nn.functional as F
from torch.autograd import Variable

import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

import torchvision.models as models

import os
import sys
import math

import numpy as np
import matplotlib.pyplot as plt

img_path = '/media/zfq/本地磁盘/cataract/train/train_img/train_img'
all_img_names = os.listdir(img_path)

data = []
for i in range(len(all_img_names)):
    img_name = all_img_names[i]
    image = io.imread(img_path + '/' + img_name)
    data = np.concatenate(data, image, axis=0)
    
data = data.astype(np.float32)/255.

means = []
stdevs = []
for i in range(3):
    pixels = data[:,:,:,i].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
