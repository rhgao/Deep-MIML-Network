#!/usr/bin/env python
# -*- coding: utf-8 -*-

from .base_options import BaseOptions

class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--which_epoch', type=str, default='latest', help='which epoch to load?')
		self.parser.add_argument('--how_many', type=int, default=50, help='how many test images to run')
		self.mode = 'test'
		self.isTrain = False
