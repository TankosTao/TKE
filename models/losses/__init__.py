# -*- coding: utf-8 -*-

import copy
from .DKE_loss import SEG_e2ec,SEG_e2ec_dml

__all__ = ['build_loss']
support_loss = ['SEG_e2ec','SEG_e2ec_dml']

def build_loss(config):
    copy_config = copy.deepcopy(config)
    loss_type = copy_config.pop('type')
    assert loss_type in support_loss, f'all support loss is {support_loss}'
    criterion = eval(loss_type)(**copy_config)
    return criterion
