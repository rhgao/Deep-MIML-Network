#!/usr/bin/env python
# -*- coding: utf-8 -*-

from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import time

opt = TestOptions().parse()
opt.batchSize = 1  # set batchSize = 1 for testing

data_loader = CreateDataLoader(opt)
dataset = data_loader.load_data()
dataset_size = len(data_loader)
print('#testing images = %d' % dataset_size)

model = torch.load(os.path.join('.', opt.checkpoints_dir, opt.name, str(opt.which_epoch) + '.pth'))
model.BasesNet.eval()

accuracies = []
losses = []

for i, data in enumerate(dataset):
	if i >= opt.how_many:
		break
	print(i)
	accuracy, loss = model.test(data)
	accuracies.append(accuracy)
	losses.append(loss)

accuracy = sum(accuracies)/len(accuracies)
loss = sum(losses)/len(losses)

print(opt.mode + ' accuracy is: ' + str(accuracy))
print(opt.mode + ' loss is: ' + str(loss))