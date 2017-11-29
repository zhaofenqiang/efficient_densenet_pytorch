#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:06:55 2017

@author: zfq
"""
from __future__ import print_function, division
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from models import CataractDenseNet
import torchvision as tv
from torch.autograd import Variable
from PIL import Image
import csv 
from pandas import read_csv
from sklearn.metrics import roc_curve, auc
from math import isnan
from torch import nn, optim
#%%
growth_rate=24
block_config=[6, 12, 32, 32]
n_epochs=10
batch_size=8
base_lr=0.05
wd=0.0001
momentum=0.9
num_init_features=64
num_tools = 21

truth_lable_name = '/media/xnat/软件/cataract/train/valid.csv'
prediction_label_name = '/media/xnat/软件/cataract/train/cataract_densenet169_12_Predicted.csv'

#%%
img_path = '/media/xnat/软件/cataract/train/valid_img'
all_img_names = os.listdir(img_path)

mean = [0.61372, 0.34675, 0.14977]
stdv = [0.18669, 0.17454, 0.11950]
valid_transforms = tv.transforms.Compose([
    tv.transforms.ToTensor(),
    tv.transforms.Normalize(mean=mean, std=stdv),
])   
   

#model = tv.models.resnet50()
#model.fc = nn.Linear(model.fc.in_features, 21)

model = CataractDenseNet(
        growth_rate = growth_rate,
        block_config = block_config,
        num_classes = 21,
        num_init_features = num_init_features,
    )

print(model)
model_cuda = model.cuda()
model_cuda.load_state_dict(torch.load('/home/xnat/pytorch_densenet_cataract/tmp/cataract_densenet169_12'))
model_cuda.eval()

with open("/media/xnat/软件/cataract/train/cataract_densenet169_12_Predicted.csv","w") as csvfile: 
    writer = csv.writer(csvfile)
    for i in range(len(all_img_names)):
        img_name = all_img_names[i]
        image = Image.open(img_path + '/' + img_name)
        image = valid_transforms(image)
        image.unsqueeze_(0)
        input_var = Variable(image, volatile=True).cuda()
        predict_labels = model_cuda(input_var)
#        predict_labels[(predict_labels <= 0)] = 0
#        predict_labels[(predict_labels > 0)] = 1
        row =[]
        row.append(img_name)
        row.extend(predict_labels.data[0])
        writer.writerow(row) 
        print(i)
#%%


def auc_tool(truth_filename, prediction_filename, tool):

    truth = []
    predictions = []

    #parsing the right column for the current tool
    truth_data = read_csv(truth_filename, header = 0, skipinitialspace = True,
				usecols = [tool], squeeze = True, dtype = 'float32').tolist()
    prediction_data = read_csv(prediction_filename, header = None, skipinitialspace = True,
									usecols = [tool], squeeze = True, dtype = 'float32').tolist()
    if len(truth_data) != len(prediction_data):
        raise ValueError('Files {} and {} have different row counts'.
							format(truth_filename, prediction_filename))
	   # appending rows with consensual ground truth
    indices = [index for index, value in enumerate(truth_data) if value != 0.5 ]
    truth += [truth_data[index] for index in indices]
    predictions += [prediction_data[index] for index in indices]

		# computing the area under the ROC curve
    fpr, tpr, _ = roc_curve(truth, predictions)
    score = auc(fpr, tpr)
    return 0. if isnan(score) else score
    return 0.


scores = []
for tool in range(1, num_tools + 1):
    score = auc_tool(truth_lable_name, prediction_label_name, tool)
    print('Score tool {0}: {1:.4f}'.format(tool, score))
    scores.append(score)
# computing the average score
print('Average: {0:.4f}'.format(sum(scores) / float(len(scores))))



