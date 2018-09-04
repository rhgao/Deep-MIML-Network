#!/usr/bin/env python
# -*- coding: utf-8 -*-

def create_model(opt):
    model = None
    print(opt.model)
    if opt.model == 'MIML':
        from .MIML import MIMLModel
        model = MIMLModel()
    elif opt.model == 'KL_divergence':
        from .KL_divergence import KLModel
        model = KLModel()
    else:
        raise ValueError("Model [%s] not recognized." % opt.model)
    model.initialize(opt)
    print("model [%s] was created" % (model.name()))
    return model

