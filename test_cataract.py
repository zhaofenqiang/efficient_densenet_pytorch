#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:06:55 2017

@author: zfq
"""
import os
from math import isnan
from pandas import read_csv
from sklearn.metrics import roc_curve, auc
import torch
from models import CataractDenseNet

#%%
num_tools = 21
num_files = 25	
growth_rate = 12
block_config=[6, 24, 20, 16]
batch_size = 8
truth_filename = '/media/zfq/本地磁盘/cataract/train/train_labels/train01.csv'
prediction_filename = '/media/zfq/本地磁盘/cataract/train/train_labels/train01.csv'

def auc_tool(truth_filename, prediction_filename, tool):

    truth = []
    predictions = []

    # loop on (truth, predictions) file pairs
    for file in range(1, num_files + 1):

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

def test(truth_filename, prediction_filename):
    scores = []
    for tool in range(1, num_tools + 1):
        score = auc_tool(truth_filename, prediction_filename, tool)
        print('Score tool {0}: {1:.4f}'.format(tool, score))
        scores.append(score)
    # computing the average score
    print('Average: {0:.4f}'.format(sum(scores) / float(len(scores))))

        
        
model = CataractDenseNet(
        growth_rate = growth_rate,
        block_config=[6, 24, 20, 16],
        num_classes=21,
    )
print(model)
model_cuda = model.cuda()
model_cuda.load_state_dict(torch.load('tmp/cataract_model_10'))



