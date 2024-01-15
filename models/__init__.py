# -*- coding: utf-8 -*-

import copy
from .model import DKE
from .losses import build_loss



__all__ = ['build_loss', 'build_model','build_decode']
support_model = ['DKE']


def build_model(config):
    """
    get architecture model class
    """
    copy_config = copy.deepcopy(config)
    arch_type = copy_config.pop('type')
    assert arch_type in support_model, f'{arch_type} is not developed yet!, only {support_model} are support now'
    arch_model = eval(arch_type)(copy_config)
    return arch_model

