# -*- coding: utf-8 -*-
from .FPN import FPN


__all__ = ['build_neck']
support_neck = ['FPN']


def build_neck(neck_name, **kwargs):
    print(kwargs)
    assert neck_name in support_neck, f'all support neck is {support_neck}'
    neck = eval(neck_name)(**kwargs)
    return neck
