#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import os
from util import util
import torch

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--K', type=int, default=4, help='number of sub-concepts in the  sub-concept layer')
		self.parser.add_argument('--L', type=int, default=1000, help='number of classes in the label space')
		self.parser.add_argument('--F', type=int, default=2401, help='dimension of basis vectors')
		self.parser.add_argument('--num_of_bases', type=int, default=25, help='number of basis vectors for one audio')
		self.parser.add_argument('--num_of_fc', type=int, default=0, help='num of fc layers before the sub-concept layer')
		self.parser.add_argument('--with_batchnorm', action='store_true', help='use batchnorm in basesnet')
		self.parser.add_argument('--fc_dimension', type=int, default=1024, help='dimension of fully-connected layers')
		self.parser.add_argument('--HDF5FileRoot', required=True, help='path to the folder that contains train.h5, val.h5 and test.h5')
		self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')
		self.parser.add_argument('--name', type=str, default='audioVisual', help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints', help='models are saved here')
		self.parser.add_argument('--model', choices=('MIML','KL_divergence'), default='MIML', help='choose which model to use and how datasets are loaded. [MIML]')
		self.parser.add_argument('--batchSize', type=int, default=512, help='input batch size')
		self.parser.add_argument('--nThreads', default=2, type=int, help='# threads for loading data')
		self.parser.add_argument('--norm', choices=('l1','l2','max','none'), default='max', help='type of normalization on the basis vectors ')
		self.parser.add_argument('--dataset', choices=('musicInstruments','animals','vehicles','all'), default='musicInstruments', help='data to use')
		self.parser.add_argument('--using_multi_labels', action='store_true', help='whether to use multi-labels')
		self.parser.add_argument('--multi_label_threshold', type=float, default=0.3, help='above this threshhold it will be assigned as gt label')		
		self.parser.add_argument('--selected_classes', action='store_true', help='whether to take only related classes from ImageNet predictions')
		self.parser.add_argument('--zeroCenterInput', action='store_true', help='whether to make the input to the network zero centered')		
		self.initialized = True

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.isTrain = self.isTrain # train or test
		self.opt.mode = self.mode

		#parse gpu ids
		str_ids = self.opt.gpu_ids.split(',')
		self.opt.gpu_ids = []
		for str_id in str_ids:
			id = int(str_id)
			if id >= 0:
				self.opt.gpu_ids.append(id)

		# set gpu ids
		if len(self.opt.gpu_ids) > 0:
			torch.cuda.set_device(self.opt.gpu_ids[0])

		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')

		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		util.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
