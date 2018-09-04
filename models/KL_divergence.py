#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable

class KLModel(BaseModel):
    def name(self):
        return 'KLModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        self.label = self.Tensor(opt.batchSize)
        self.bases = self.Tensor(opt.batchSize, opt.F, opt.num_of_bases)

        self.BasesNet = networks.BasesNet(opt)
        self.sub_concept_pooling = nn.modules.MaxPool2d((opt.K, 1), stride=(1,1))
        self.instance_pooling = nn.modules.MaxPool2d((opt.num_of_bases,1), stride=(1,1))
        self.softmax = nn.LogSoftmax(dim=-1)
        self.loss = nn.KLDivLoss()


        if(len(opt.gpu_ids)>0):
            self.BasesNet.cuda(opt.gpu_ids[0])
            self.sub_concept_pooling.cuda(opt.gpu_ids[0])
            self.instance_pooling.cuda(opt.gpu_ids[0])
            self.softmax.cuda(opt.gpu_ids[0])
            self.loss.cuda(opt.gpu_ids[0])

        if self.isTrain and not opt.continue_train:
            networks.init_weights(self.BasesNet, self.opt.init_type)
        elif self.isTrain :
            print('loading epoch', opt.epoch_count, 'model to continue training!')
            self.load_network(self.BasesNet, opt.epoch_count)
        else:
            print('loading model to test!')
            self.load_network(self.BasesNet, opt.which_epoch)

        if(self.isTrain):
            self.optimizer = optim.Adam(list(self.BasesNet.parameters()), lr=opt.learning_rate, weight_decay=0.00001)
        else:
            self.BasesNet.eval()

        self.batch_loss = []

    def forward(self, input, volatile=False):
        bases = input['bases'].unsqueeze(3) #add another dimension for 2D convolution, a trick to replace fc with 1x1conv
        label = input['label']
        self.bases.resize_(bases.size()).copy_(bases)
        self.label.resize_(label.size()).copy_(label)

        #print(self.bases.size())
        # shape: (batchSize, L*K, num_of_bases)
        basesnet_output = self.BasesNet(Variable(self.bases)).view(-1, self.opt.L, self.opt.K, self.opt.num_of_bases)
        #print("sub_concept_layer_output:",basesnet_output.size())
        # shape: (batchSize, L, K, num_of_bases)
        sub_concept_pooling_output = self.sub_concept_pooling(basesnet_output).view(-1, self.opt.L, self.opt.num_of_bases).permute(0,2,1).unsqueeze(1)
        #print("sub_concept_pooling_output:",sub_concept_pooling_output.size())
        # shape:  (batchSize, 1, 1, L)
        self.output = self.softmax(self.instance_pooling(sub_concept_pooling_output).view(-1, self.opt.L))
        #print("final_output:", self.output.size())
        # shape -> (batchSize, L)

    def decrease_learning_rate(self, times, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.opt.learning_rate * pow(factor, times)
        print("current learning rate:",self.opt.learning_rate * pow(factor, times))

    def backward(self):
        label = Variable(self.label, requires_grad=False)
        loss = self.loss(self.output, label)
        self.batch_loss.append(loss.data[0])
        loss.backward()

    def display_train(self, writer, index):
        loss = sum(self.batch_loss)/len(self.batch_loss)
        writer.add_scalar('data/loss', loss, index)
        print('loss: ' + str(loss))
        self.batch_loss = []

    def display_val(self, writer, index, dataset_val):
        losses = []
        for i, val_data in enumerate(dataset_val):
            if i >= self.opt.validation_batches:
                break
            loss = self.test(val_data)
            losses.append(loss)
        loss = sum(losses)/len(losses)
        writer.add_scalar('data/val_loss', loss, index)
        print('validation loss is: ' + str(loss))

    def test(self, input):
        self.forward(input, volatile=True)
        label = Variable(self.label, requires_grad=False)
        loss = self.loss(self.output, label).data.cpu().numpy()[0]
        return loss
        
    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()

    def save(self, index):
        self.save_network(self.BasesNet, index, self.gpu_ids)