#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
from data.base_dataset import BaseDataset
import h5py
import numpy as np
import sklearn.preprocessing
import torchvision.transforms as transforms
import torch

def normalizeBases(bases, norm):
	if norm == "l1":
		return sklearn.preprocessing.normalize(bases, norm='l1', axis=0)
	elif norm == "l2":
		return sklearn.preprocessing.normalize(bases, norm='l2', axis=0)
	elif norm == "max":
		return sklearn.preprocessing.normalize(bases, norm='max', axis=0)
	else:
		return bases

class KLDataset(BaseDataset):
	def initialize(self, opt):
		self.opt = opt
		self.bases = []

		#load hdf5 file here
		h5f_path = os.path.join(opt.HDF5FileRoot, opt.mode + ".h5")
		h5f = h5py.File(h5f_path, 'r')
		self.bases = h5f['bases'][:]
		self.labels = h5f['labels'][:]
		
	def __getitem__(self, index):
		bases = np.load(self.bases[index].decode("utf-8"))
		label = np.load(self.labels[index].decode("utf-8"))

		#perform basis normalization
		bases = normalizeBases(bases, self.opt.norm)
		return {'bases': bases, 'label': label}

	def __len__(self):
		return len(self.bases)

	def name(self):
		return 'KLDataset'