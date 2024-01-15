# -*- coding: utf-8 -*-

from __future__ import print_function

import argparse
import os

import anyconfig


def init_args():
    parser = argparse.ArgumentParser(description='DKE.pytorch')
    parser.add_argument('config_file', default='', type=str)
    parser.add_argument('--local_rank', dest='local_rank', default=0, type=int, help='Use distributed training')
    parser.add_argument('--alpha', default=1, type=float, help='')
    parser.add_argument('--beta', default=0.25, type=float, help='')
    parser.add_argument('--gama', default=0.25, type=float, help='')
    parser.add_argument('--output_dir', default='output/', type=str, help='')
    args = parser.parse_args()
    return args


def main(config):
    import torch
    from models import build_model, build_loss
    from data_loader import get_dataloader
    from trainer import Trainer
    from utils import get_metric
    
    if torch.cuda.device_count() > 1:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method="env://", world_size=torch.cuda.device_count(), rank=args.local_rank)
        config['distributed'] = True
    else:
        config['distributed'] = False
    config['local_rank'] = args.local_rank  
    train_loader = get_dataloader(config['dataset']['train'], config['distributed'])
    assert train_loader is not None
    if 'validate' in config['dataset']:
        validate_loader = get_dataloader(config['dataset']['validate'], False)
    else:
        validate_loader = None
    config['loss']['alpha'] = args.alpha
    config['loss']['beta'] = args.beta
    config['loss']['gama'] = args.gama
    config['trainer']['output_dir'] = args.output_dir
    criterion = build_loss(config['loss']).cuda()

    config['arch']['backbone']['in_channels'] = 3 if config['dataset']['train']['dataset']['args']['img_mode'] != 'GRAY' else 1
    model = build_model(config['arch'])
    metric = get_metric(config['metric'])
    print('starting train')
    trainer = Trainer(config=config,
                      model=model,
                      criterion=criterion,
                      train_loader=train_loader,
                      post_process=None,
                      metric_cls=metric,
                      validate_loader=validate_loader)
    trainer.train()

def train_main():
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))

    from utils import parse_config

    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)

    
if __name__ == '__main__':
    # train_main()
    import sys
    import pathlib
    __dir__ = pathlib.Path(os.path.abspath(__file__))
    sys.path.append(str(__dir__))
    sys.path.append(str(__dir__.parent.parent))
    from utils import parse_config
    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    main(config)
    
