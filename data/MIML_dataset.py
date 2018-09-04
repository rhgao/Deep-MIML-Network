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

def subsetOfClasses(label):
	#15 music instruments: accordion, acoustic_guitar, banjo, cello, drum, electric_guitar, flute, french_horn, harmonica, harp, marimba, piano, saxophone, trombone, violin
	indexes = [[401], [402], [420], [486], [541], [546], [558], [566], [593], [594], [642], [579,881], [776], [875], [889]]
	selected_label = np.zeros(15)
	for i,class_indexes in enumerate(indexes):
		for index in class_indexes:
			selected_label[i] = selected_label[i] + label[index]
		selected_label[i] = selected_label[i] / len(class_indexes)
	return selected_label

def subsetOfClassesAnimals(label):
	#4 animals: cat, dog, chicken, frog
	#start -> end
	indexes = [[281,285],[151,275],[7,8],[30,32]]
	selected_label = np.zeros(4)
	for i,indexlist in enumerate(indexes):
		start = indexlist[0]
		end = indexlist[1]
		for j in range(start, end+1):
			selected_label[i] = selected_label[i] + label[j]
		selected_label[i] = selected_label[i] / (end - start + 1)
	return selected_label

def subsetOfClassesVehicles(label):
	#4 vehicles: racing_car, train, plane, motor_scooter
	indexes = [[751,817,511], [705,466,820,547], [726,404,895], [670,665]]
	selected_label = np.zeros(4)
	for i,class_indexes in enumerate(indexes):
		for index in class_indexes:
			selected_label[i] = selected_label[i] + label[index]
		selected_label[i] = selected_label[i] / len(class_indexes)
	return selected_label

def subsetOfClassesAll(label):
	selected_label = np.zeros(23)
	#animals
	indexes = [[281,285],[151,275],[7,8],[30,32]]
	for i,indexlist in enumerate(indexes):
		start = indexlist[0]
		end = indexlist[1]
		for j in range(start, end+1):
			selected_label[i] = selected_label[i] + label[j]
		selected_label[i] = selected_label[i] / (end - start + 1)
	#musical instruments
	indexes = [[401], [402], [420], [486], [541], [546], [558], [566], [593], [594], [642], [579,881], [776], [875], [889]]
	for i,class_indexes in enumerate(indexes):
		for index in class_indexes:
			selected_label[i+4] = selected_label[i+4] + label[index]
		selected_label[i+4] = selected_label[i+4] / len(class_indexes)
	#vehicles
	indexes = [[751,817,511], [705,466,820,547], [726,404,895], [670,665]]
	for i,class_indexes in enumerate(indexes):
		for index in class_indexes:
			selected_label[i+19] = selected_label[i+19] + label[index]
		selected_label[i+19] = selected_label[i+19] / len(class_indexes)	
	return selected_label

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)

class MIMLDataset(BaseDataset):
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
		if self.opt.selected_classes:
			if self.opt.dataset == 'musicInstruments':
				loaded_label = softmax(subsetOfClasses(np.load(self.labels[index].decode("utf-8"))))
			elif self.opt.dataset == 'animals':
				loaded_label = softmax(subsetOfClassesAnimals(np.load(self.labels[index].decode("utf-8"))))
			elif self.opt.dataset == 'vehicles':
				loaded_label = softmax(subsetOfClassesVehicles(np.load(self.labels[index].decode("utf-8"))))
			elif self.opt.dataset == 'all':
				loaded_label = softmax(subsetOfClassesAll(np.load(self.labels[index].decode("utf-8"))))
		else:
			loaded_label = softmax(np.load(self.labels[index].decode("utf-8")))

		if self.opt.using_multi_labels:
			label = np.zeros(self.opt.L) - 1 #-1 means incorrect labels
			label_index = [np.argmax(loaded_label)]
			label_index = list(set(label_index) | set(np.where(loaded_label >= 0.3)[0]))
			for i in range(len(label_index)):
				label[i] = label_index[i]
		else:
			label = np.argmax(loaded_label)

		#perform basis normalization
		bases = normalizeBases(bases, self.opt.norm)
		if self.opt.zeroCenterInput:
			bases = bases * 2 - 1

		if self.opt.isTrain:
			return {'bases': bases, 'label': label}
		else:
			return {'bases': bases, 'label': label}

	def __len__(self):
		return len(self.bases)

	def name(self):
		return 'MIMLDataset'