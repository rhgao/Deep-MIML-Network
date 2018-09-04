#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from options.train_options import TrainOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
from tensorboardX import SummaryWriter
import torch

opt = TrainOptions().parse()
data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#training images = %d' % dataset_size)

#create validation set data loader if validation_on option is set
if opt.validation_on:
        #temperally set to val to load val data
        opt.mode = 'val'
        data_loader_val = CreateDataLoader(opt)
        dataset_val = data_loader_val.load_data()
        dataset_size_val = len(data_loader_val)
        print('#validation images = %d' % dataset_size_val)
        opt.mode = 'train' #set it back

writer = SummaryWriter(comment=opt.name)
total_steps = 0

data_loading_time = []
model_forward_time = []
model_backward_time = []

model = create_model(opt)

#if continue train find the current largest epoch number and resume, set lr to appropriate number
if opt.continue_train:
        checkpoints = os.listdir(os.path.join('.', opt.checkpoints_dir, opt.name))
        current_max_epoch_saved = 0
        for checkpoint in checkpoints:
                if checkpoint.endswith('.pth') and not checkpoint.startswith('latest'):
                        epoch_number = int(checkpoint[:-4])
                        if epoch_number > current_max_epoch_saved:
                                current_max_epoch_saved = epoch_number
        print("starting from epoch ", current_max_epoch_saved)
        opt.epoch_count = current_max_epoch_saved
        total_steps = opt.epoch_count * opt.batchSize
        #if no saved models found, start from scratch 
        if opt.epoch_count == 0:
                opt.continue_train = False
        else:
                model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, str(opt.epoch_count) + '.pth'))

if(opt.continue_train and opt.learning_rate_decrease_itr > 0):
        # set correct starting lr to resume training
        model.decrease_learning_rate(current_max_epoch_saved // opt.learning_rate_decrease_itr, opt.decay_factor) 

for epoch in range(1 + opt.epoch_count, opt.niter+1):
        epoch_start_time = time.time()

        if(opt.measure_time):
                iter_start_time = time.time()

        for i, data in enumerate(dataset):
                if(opt.measure_time):
        	        iter_data_loaded_time = time.time()

                total_steps += opt.batchSize
                
                model.forward(data)
                
                if(opt.measure_time):
        	        iter_data_forwarded_time = time.time()

                model.optimize_parameters()

                if(opt.measure_time):
                        iter_model_backwarded_time = time.time()
                        data_loading_time.append(iter_data_loaded_time - iter_start_time)
                        model_forward_time.append(iter_data_forwarded_time - iter_data_loaded_time)
                        model_backward_time.append(iter_model_backwarded_time - iter_data_forwarded_time)

                if(total_steps // opt.batchSize % opt.display_freq == 0):
                        print('Display training progress at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        model.display_train(writer, total_steps)
                        if(opt.measure_time):
                                print('average data loading time: ' + str(sum(data_loading_time)/len(data_loading_time)))
                                print('average forward time: ' + str(sum(model_forward_time)/len(model_forward_time)))
                                print('average backward time: ' + str(sum(model_backward_time)/len(model_backward_time)))
                                data_loading_time = []
                                model_forward_time = []
                                model_backward_time = []
                        print('end of display \n')

                if(total_steps // opt.batchSize % opt.save_latest_freq == 0):
                        print('saving the latest model (epoch %d, total_steps %d)' % (epoch, total_steps))
                        torch.save(model, os.path.join('.', opt.checkpoints_dir, opt.name, 'latest.pth'))

                if(total_steps // opt.batchSize % opt.validation_freq == 0 and opt.validation_on):
                        model.BasesNet.eval()
                        print('Display validation results at (epoch %d, total_steps %d)' % (epoch, total_steps))
                        model.display_val(writer, total_steps, dataset_val)
                        print('end of display \n')
                        model.BasesNet.train()

                if(opt.measure_time):
                        iter_start_time = time.time()

        if(epoch % opt.save_epoch_freq == 0):
                print('saving the model at the end of epoch %d, total_steps %d' % (epoch, total_steps))
                torch.save(model, os.path.join('.', opt.checkpoints_dir, opt.name, str(epoch) + '.pth'))
                torch.save(model, os.path.join('.', opt.checkpoints_dir, opt.name, 'latest.pth'))

        #decrease learning rate 6% every opt.learning_rate_decrease_itr epoches
        if(opt.learning_rate_decrease_itr > 0 and epoch % opt.learning_rate_decrease_itr == 0):
                model.decrease_learning_rate(epoch / opt.learning_rate_decrease_itr, opt.decay_factor)
                print('decreased learning rate by ', opt.decay_factor)
