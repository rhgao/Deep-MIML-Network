#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
from torch.nn import init

def weights_init_normal(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('Linear') != -1:
        init.normal(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_xavier(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('Linear') != -1:
        init.xavier_normal(m.weight.data, gain=0.02)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def weights_init_orthogonal(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('Linear') != -1:
        init.orthogonal(m.weight.data, gain=1)
    elif classname.find('BatchNorm2d') != -1:
        init.normal(m.weight.data, 1.0, 0.02)
        init.constant(m.bias.data, 0.0)

def init_weights(net, init_type='normal'):
    print('initialization method [%s]' % init_type)
    if init_type == 'normal':
        net.apply(weights_init_normal)
    elif init_type == 'xavier':
        net.apply(weights_init_xavier)
    elif init_type == 'kaiming':
        net.apply(weights_init_kaiming)
    elif init_type == 'orthogonal':
        net.apply(weights_init_orthogonal)
    else:
        raise NotImplementedError('initialization method [%s] is not implemented' % init_type)

class BasesNet(nn.Module):
    def __init__(self, opt):
        super(BasesNet, self).__init__()
        self.gpu_ids = opt.gpu_ids
        model = []
        # shape -> (batchSize, bases_dimension, num_of_bases, 1)
        if opt.num_of_fc > 0:
            model = [nn.modules.Conv2d(opt.F, opt.fc_dimension, 1, 1)]
            if opt.with_batchnorm:
                model.append(nn.BatchNorm2d(opt.fc_dimension))
            model.append(nn.ReLU())
            for i in range(opt.num_of_fc - 1):
                model.append(nn.modules.Conv2d(opt.fc_dimension, opt.fc_dimension, 1, 1))
                if opt.with_batchnorm:
                    model.append(nn.BatchNorm2d(opt.fc_dimension))
                model.append(nn.ReLU())
            self.sub_concept_layer = nn.modules.Conv2d(opt.fc_dimension, opt.L * opt.K, 1, 1)
            model.append(self.sub_concept_layer)
            # shape -> (batchSize, L*K, num_of_bases)
            if opt.with_batchnorm:
                model.append(nn.BatchNorm2d(opt.L * opt.K))
            model.append(nn.ReLU())
        else:
            self.sub_concept_layer = nn.modules.Conv2d(opt.F, opt.L * opt.K, 1, 1)
            # shape -> (batchSize, L*K, num_of_bases)
            if opt.with_batchnorm:
                model = [self.sub_concept_layer, nn.BatchNorm2d(opt.L * opt.K), nn.ReLU()]
            else:
                model = [self.sub_concept_layer, nn.ReLU()]
        self.model = nn.Sequential(*model)
    
    def forward(self, input):
        if self.gpu_ids and isinstance(input.data, torch.cuda.FloatTensor):
            return nn.parallel.data_parallel(self.model, input, self.gpu_ids)
        else:
            return self.model(input)

