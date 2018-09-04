#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

class BaseModel():
    def name(self):
        return 'BaseModel'

    def initialize(self, opt):
        self.opt = opt
        self.gpu_ids = opt.gpu_ids
        self.isTrain = opt.isTrain
        self.Tensor = torch.cuda.FloatTensor if self.gpu_ids else torch.Tensor
        self.LabelTensor = torch.cuda.LongTensor if self.gpu_ids else torch.LongTensor
        self.save_dir = os.path.join(opt.checkpoints_dir, opt.name)

    def set_input(self, input):
        self.input = input

    def forward(self):
        pass

    # used in test time, no backprop
    def test(self):
        pass

    def optimize_parameters(self):
        pass

    def get_current_errors(self):
        return {}

    def save(self, label):
        pass