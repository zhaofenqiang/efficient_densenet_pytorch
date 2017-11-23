#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 20:41:03 2017
@author: zfq
"""
from skimage import io
import os
import numpy as np
import matplotlib.pyplot as plt
#%%
img_path = '/media/zfq/本地磁盘/cataract/train/train_img'
all_img_names = os.listdir(img_path)

img_name = all_img_names[0]
image = io.imread(img_path + '/' + img_name)
image = image.astype(np.float32)/255.
data =  image[np.newaxis, :]

for i in range(1, len(all_img_names)):
    if i % 120 == 0 :
        img_name = all_img_names[i-60]
        image = io.imread(img_path + '/' + img_name)
        image = image.astype(np.float32)/255.
        data = np.concatenate((data, image[np.newaxis, :]), axis=0)
        print i

means = []
stdevs = []
for i in range(3):
    pixels = data[:,:,:,i].ravel()
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))

# means:                   std:   
