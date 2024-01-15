# -*- coding: utf-8 -*-
from ast import arg
from email.policy import strict
import os
import sys
import pathlib
__dir__ = pathlib.Path(os.path.abspath(__file__))
sys.path.append(str(__dir__))
sys.path.append(str(__dir__.parent.parent))
from utils.result_format import ResultFormat
import argparse
import torch
from tqdm.auto import tqdm
import time
class EVAL():
    def __init__(self, args,config=None, gpu_id=0):
        from models import build_model
        from data_loader import get_dataloader
        from utils import get_metric
        self.gpu_id = gpu_id
        self.args = args
        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        model_path = args.model_path
        checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
        if config is None:
            config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False
        self.validate_loader = get_dataloader(config['dataset']['validate'], False)
        self.model = build_model(config['arch'])
        self.model.load_state_dict(checkpoint['state_dict'],strict =False)
        self.model.to(self.device)

        self.metric_cls = get_metric(config['metric'])
        self.fps = []
    
    def deal_result(self,metrics):
        print('recall',metrics['recall'].avg,'precision', metrics['precision'].avg,'fmeasure', metrics['fmeasure'].avg)
    def eval(self):
        is_save_result = True
        self.model.eval()
        torch.cuda.empty_cache()  # speed up evaluating after training finished
        metric = {'coarse':[],'pred':[]}
        total_frame = 0.0
        total_time = 0.0
        if is_save_result:
            rf = ResultFormat("TT", 'outputs/submit_{}'.format(self.args.dataset))
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                batch['img']  = batch['img'].to(self.device)
                start = time.time()
                output = self.model(batch['img'],batch,is_training = False)
                if not args.pred: 
                    boxes = output['coarse_polys']
                else:
                    boxes = output['pred_polys']
                if is_save_result:
                    if len(boxes)>0:
                        rf.write_result(batch['img_name'][0], boxes[0])
                    else:
                        rf.write_result(batch['img_name'][0], [])
                
                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                metric['coarse'].append(self.metric_cls.validate_measure(batch, output['coarse_polys']))
                if "pred_polys" in output.keys():
                    metric['pred'].append(self.metric_cls.validate_measure(batch, output['pred_polys']))
        metrics = {'coarse':self.metric_cls.gather_measure(metric['coarse'])}
        self.deal_result(metrics['coarse'])
        if "pred_polys" in output.keys():
            metrics.update({'pred':self.metric_cls.gather_measure(metric['pred'])})
            self.deal_result(metrics['pred'])

def init_args():
    parser = argparse.ArgumentParser(description='DKE.pytorch')
    parser.add_argument('config_file',default='', type=str)
    parser.add_argument('--model_path', required=False,default='', type=str)
    parser.add_argument('--pred',action='store_true')
    parser.add_argument('--dataset', default='total',type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = init_args()
    from utils import parse_config
    import anyconfig
    args = init_args()
    assert os.path.exists(args.config_file)
    config = anyconfig.load(open(args.config_file, 'rb'))
    if 'base' in config:
        config = parse_config(config)
    eval = EVAL(args,config)
    result = eval.eval()

