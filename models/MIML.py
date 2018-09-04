#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import torch
from torch import optim
import torch.nn as nn
from .base_model import BaseModel
from . import networks
from torch.autograd import Variable
from sklearn.metrics import average_precision_score

class MIMLModel(BaseModel):
    def name(self):
        return 'MIMLModel'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)
        self.isTrain = opt.isTrain

        if opt.using_multi_labels:
            self.label = self.Tensor(opt.batchSize, opt.L)
        else:
            self.label = self.Tensor(opt.batchSize)
        self.bases = self.Tensor(opt.batchSize, opt.F, opt.num_of_bases)

        self.BasesNet = networks.BasesNet(opt)
        self.sub_concept_pooling = nn.modules.MaxPool2d((opt.K, 1), stride=(1,1))
        self.instance_pooling = nn.modules.MaxPool2d((opt.num_of_bases,1), stride=(1,1))
        self.softmax = nn.Softmax(dim=-1)
        self.sigmoid = nn.Sigmoid()

        if opt.using_multi_labels:
            self.loss = nn.MultiLabelMarginLoss()
        else:
            self.loss = nn.CrossEntropyLoss()

        if(len(opt.gpu_ids)>0):
            self.BasesNet.cuda(opt.gpu_ids[0])
            self.sub_concept_pooling.cuda(opt.gpu_ids[0])
            self.instance_pooling.cuda(opt.gpu_ids[0])
            self.softmax.cuda(opt.gpu_ids[0])
            self.sigmoid.cuda(opt.gpu_ids[0])
            self.loss.cuda(opt.gpu_ids[0])

        networks.init_weights(self.BasesNet, self.opt.init_type)

        if(self.isTrain):
            if opt.using_multi_labels:
                self.optimizer = optim.Adam(list(self.BasesNet.parameters()), lr=opt.learning_rate, weight_decay=0.00001)
            else:
                self.optimizer = optim.Adam(list(self.BasesNet.parameters()), lr=opt.learning_rate, weight_decay=0.00001)
        else:
            self.BasesNet.eval()

        self.batch_loss = []
        self.batch_accuracy = []
        self.batch_ap = []

    def forward(self, input, volatile=False):
        bases = input['bases'].unsqueeze(3) #add another dimension for 2D convolution, a trick to replace fc with 1x1conv
        label = input['label']
        self.bases.resize_(bases.size()).copy_(bases)
        self.label.resize_(label.size()).copy_(label)

        #print(self.bases.size())
        # shape: (batchSize, L*K, num_of_bases)
        basesnet_output = self.BasesNet(Variable(self.bases, requires_grad=False, volatile=volatile)).view(-1, self.opt.L, self.opt.K, self.opt.num_of_bases)
        #print("sub_concept_layer_output:",basesnet_output.size())
        # shape: (batchSize, L, K, num_of_bases)
        sub_concept_pooling_output = self.sub_concept_pooling(basesnet_output).view(-1, self.opt.L, self.opt.num_of_bases).permute(0,2,1).unsqueeze(1)
        #print("sub_concept_pooling_output:",sub_concept_pooling_output.size())
        #softmax
        if self.opt.with_softmax:
            softmax_normalization_output = self.softmax(sub_concept_pooling_output)
            self.output = self.instance_pooling(softmax_normalization_output).view(-1, self.opt.L)
        else:
            self.output = self.instance_pooling(sub_concept_pooling_output).view(-1, self.opt.L)

    def getInstanceLabelRelation(self, input, volatile=True):
        bases = input['bases'].unsqueeze(3) #add another dimension for 2D convolution, a trick to replace fc with 1x1conv
        label = input['label']
        self.bases.resize_(bases.size()).copy_(bases)
        self.label.resize_(label.size()).copy_(label)

        basesnet_output = self.BasesNet(Variable(self.bases, requires_grad=False, volatile=volatile)).view(-1, self.opt.L, self.opt.K, self.opt.num_of_bases)
        instanceLabelRelation = self.sub_concept_pooling(basesnet_output).view(-1, self.opt.L, self.opt.num_of_bases).permute(0,2,1)
        if self.opt.using_multi_labels:
            self.output = self.instance_pooling(instanceLabelRelation.unsqueeze(1)).view(-1, self.opt.L)
            prediction = np.zeros(self.output.size())
            gt_label = np.zeros(self.output.size())
            max_label = self.output.max(dim=1)[1].data
            for i in range(gt_label.shape[0]):
                prediction[i,max_label[i]] = 1
            prediction[self.softmax(self.output).data.cpu().numpy() >= 0.3] = 1
            for index, x in np.ndenumerate(self.label.cpu().numpy()):
                if x == -1:
                    continue
                else:
                    gt_label[index[0],int(x)] = 1
            return self.softmax(instanceLabelRelation).data.cpu().numpy(), gt_label, prediction
        else:
            if self.opt.with_softmax:
                self.output = self.instance_pooling(self.softmax(instanceLabelRelation).unsqueeze(1)).view(-1, self.opt.L)
            else:
                self.output = self.instance_pooling(instanceLabelRelation.unsqueeze(1)).view(-1, self.opt.L)
            prediction = self.output.max(dim=1)[1].data.float()
            return self.softmax(instanceLabelRelation).data.cpu().numpy(), label, prediction

    def decrease_learning_rate(self, times, factor):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.opt.learning_rate * pow(factor, times)
        print("current learning rate:",self.opt.learning_rate * pow(factor, times))

    def backward(self):
        if self.opt.using_multi_labels:
            label = Variable(self.label, requires_grad=False).long()
            prediction = np.zeros(self.output.size())
            #construt ground-truth label to compute mAP
            gt_label = np.zeros(self.output.size())
            max_label = self.output.max(dim=1)[1].data
            for i in range(gt_label.shape[0]):
                prediction[i,max_label[i]] = 1
            prediction[self.softmax(self.output).data.cpu().numpy() >= 0.3] = 1
            for index, x in np.ndenumerate(self.label.cpu().numpy()):
                if x == -1:
                    continue
                else:
                    gt_label[index[0],int(x)] = 1
            ap = average_precision_score(gt_label.T, prediction.T)
            self.batch_ap.append(ap)
        else:
            label = Variable(self.label, requires_grad=False).long()
            prediction = self.output.max(dim=1)[1].data.float()
            correct = (self.label.eq(prediction)).sum()
            accuracy = correct*1.0/self.label.size()[0]
            self.batch_accuracy.append(accuracy)
        loss = self.loss(self.output, label)
        self.batch_loss.append(loss.data[0]) 
        loss.backward()

    def display_train(self, writer, index):
        loss = sum(self.batch_loss)/len(self.batch_loss)
        writer.add_scalar('data/loss', loss, index)
        print('loss: ' + str(loss))
        self.batch_loss = []
        if self.opt.using_multi_labels:
            ap = sum(self.batch_ap)/len(self.batch_ap)
            writer.add_scalar('data/mAP', ap, index)
            print('mAP: ' + str(ap))
            self.batch_ap = []
        else:
            accuracy = sum(self.batch_accuracy)/len(self.batch_accuracy)
            writer.add_scalar('data/accuracy', accuracy, index)
            print('accuracy: ' + str(accuracy))
            self.batch_accuracy = []

    def display_val(self, writer, index, dataset_val):
        accuracies = []
        losses = []
        aps = []
        for i, val_data in enumerate(dataset_val):
            if i >= self.opt.validation_batches:
                break
            if self.opt.using_multi_labels:
                ap, loss = self.test_multi_label(val_data)
                aps.append(ap)
            else:
                accuracy,loss = self.test(val_data)
                accuracies.append(accuracy)
            losses.append(loss)
        if self.opt.using_multi_labels:
            ap = sum(aps) / len(aps)
            writer.add_scalar('data/val_mAP', ap, index)
            print('validation mAP is: ' + str(ap))
        else:
            accuracy = sum(accuracies)/len(accuracies)
            writer.add_scalar('data/val_accuracy', accuracy, index)
            print('validation accuracy is: ' + str(accuracy))
        loss = sum(losses)/len(losses)
        writer.add_scalar('data/val_loss', loss, index)
        print('validation loss is: ' + str(loss))

    def test(self, input):
        self.forward(input, volatile=True)
        prediction = self.output.max(dim=1)[1].data.float()
        correct = (self.label.eq(prediction)).sum()
        accuracy = correct*1.0/self.label.size()[0]
        label = Variable(self.label.long(), requires_grad=False)
        loss = self.loss(self.output, label).data.cpu().numpy()[0]
        return accuracy, loss

    def test_multi_label(self, input):
        self.forward(input, volatile=True)
        prediction = np.zeros(self.output.size())
        gt_label = np.zeros(self.output.size())
        max_label = self.output.max(dim=1)[1].data
        for i in range(gt_label.shape[0]):
            prediction[i,max_label[i]] = 1
        prediction[self.softmax(self.output).data.cpu().numpy() >= 0.3] = 1
        for index, x in np.ndenumerate(self.label.cpu().numpy()):
            if x == -1:
                continue
            else:
                gt_label[index[0],int(x)] = 1
        ap = average_precision_score(gt_label.T, prediction.T)
        label = Variable(self.label, requires_grad=False).long()
        loss = self.loss(self.output, label).data.cpu().numpy()[0]
        return ap, loss

    def optimize_parameters(self):
        self.optimizer.zero_grad()
        self.backward()
        self.optimizer.step()
