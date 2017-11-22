#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 21 16:34:21 2017

@author: zfq
"""
import os
import torch
import torchvision as tv
from torch.autograd import Variable
from models import DenseNetEfficient
#%%
depth=100
growth_rate=12
efficient=True
batch_size=64
if (depth - 4) % 3:
        raise Exception('Invalid depth')
block_config = [(depth - 4) // 6 for _ in range(3)]  #取整除 - 返回商的整数部分, block_config:[6, 6, 6]

model = DenseNetEfficient(
        growth_rate=growth_rate,
        block_config=block_config,
        num_classes=10,
        cifar=True
    )
model_wrapper = model.cuda(device_id=0)
model_wrapper.load_state_dict(torch.load(os.path.join('tmp', 'model_50')))

#%%
correct = 0
total = 0
mean = [0.5071, 0.4867, 0.4408]
stdv = [0.2675, 0.2565, 0.2761]

test_transforms = tv.transforms.Compose([
        tv.transforms.ToTensor(),
        tv.transforms.Normalize(mean=mean, std=stdv),
    ])
data_root = os.path.join('data', 'cifar10')

test_set = tv.datasets.CIFAR10(data_root, train=False, transform=test_transforms, download=False)
test_loader = torch.utils.data.DataLoader(test_set, pin_memory=True, batch_size=64)
for data in test_loader:
#    dataiter = iter(test_loader)
#    images, labels = dataiter.next()
    images, labels = data
    labels = Variable(labels, volatile=True).cuda(async=True)
    outputs = model(Variable(images,  volatile=True).cuda(async=True))
    _, predicted = torch.max(outputs.data, 1)
    total += labels.size(0)
    correct += (predicted == labels.data).sum()

print('Accuracy of the network on the %d test images: %d %%' % (total, 
    100 * correct / total))