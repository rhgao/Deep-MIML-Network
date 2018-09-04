#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_options import BaseOptions

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=10, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_epoch_freq', type=int, default=5, help='frequency of saving checkpoints at the end of epochs')
		self.parser.add_argument('--save_latest_freq', type=int, default=1000, help='frequency of saving the latest results')
		self.parser.add_argument('--continue_train', action='store_true', help='continue training: load the latest model')
		self.parser.add_argument('--epoch_count', type=int, default=0, help='the starting epoch count')
		self.parser.add_argument('--learning_rate', type=float, default=0.001, help='starting learning rate during training')
		self.parser.add_argument('--learning_rate_decrease_itr', type=int, default=-1, help='how often is the learning rate decreased by decay_factor')
		self.parser.add_argument('--decay_factor', type=float, default=0.94, help='decay factor for learning rate')
		self.parser.add_argument('--niter', type=int, default=300, help='# of epochs to train')
		self.parser.add_argument('--init_type', type=str, default='normal', choices=('normal','xavier','kaiming','orthogonal'), help='how the networks is initialized. Options include normal and orthogonal')
		self.parser.add_argument('--measure_time', action='store_true', help='measure time of different steps during training')
		self.parser.add_argument('--validation_on', action='store_true', help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=50, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=10, help='number of batches to test for validation')
		self.parser.add_argument('--with_softmax', action='store_true', help='whether add layerwise softmax after subconcept layer')
		self.mode = 'train'
		self.isTrain = True
