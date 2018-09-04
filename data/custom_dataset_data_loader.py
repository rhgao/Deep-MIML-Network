#!/usr/bin/env python
# -*- coding: utf-8 -*-

import torch.utils.data
from data.base_data_loader import BaseDataLoader

def CreateDataset(opt):
    dataset = None
    if opt.model == 'MIML':
        from data.MIML_dataset import MIMLDataset
        dataset = MIMLDataset()
    elif opt.model == 'KL_divergence':
        from data.KL_dataset import KLDataset
        dataset = KLDataset()
    else:
        raise ValueError("Dataset [%s] not recognized." % opt.model)

    print("dataset [%s] was created" % (dataset.name()))
    dataset.initialize(opt)
    return dataset


class CustomDatasetDataLoader(BaseDataLoader):
    def name(self):
        return 'CustomDatasetDataLoader'

    def initialize(self, opt):
        BaseDataLoader.initialize(self, opt)
        self.dataset = CreateDataset(opt)
        self.dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=opt.batchSize,
            shuffle=True,
            num_workers=int(opt.nThreads))

    def load_data(self):
        return self

    def __len__(self):
        return len(self.dataset)

    def __iter__(self):
        for i, data in enumerate(self.dataloader):
            yield data
