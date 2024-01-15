import copy
import PIL
import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.utils.data.dataloader import default_collate

def get_dataset(data_path, module_name, transform, dataset_args):
    from . import dataset

    s_dataset = getattr(dataset, module_name)(transform=transform, data_path=data_path,
                                              **dataset_args)
    return s_dataset


def get_transforms(transforms_config):
    tr_list = []
    for item in transforms_config:
        if 'args' not in item:
            args = {}
        else:
            args = item['args']
        cls = getattr(transforms, item['type'])(**args)
        tr_list.append(cls)
    tr_list = transforms.Compose(tr_list)
    return tr_list


class ICDARCollectFN:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        for sample in batch:
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (torch.Tensor)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        return data_dict
    
class ICDARCollectFCN:
    def __init__(self, *args, **kwargs):
        print('collect func11')
        pass

    def __call__(self, batch):
        data_dict = {}
        to_tensor_keys = []
        
        for sample in batch:
            # print(sample.keys())
            for k, v in sample.items():
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, torch.Tensor):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        # exit()
        return data_dict
    

class EvoCollectFN:
    def __init__(self, *args, **kwargs):
        print('collect func')
        pass
    def collect_training(self,poly, ct_01):
        ct_01 = ct_01.bool()
        batch_size = ct_01.size(0)
        poly = torch.cat([poly[i][ct_01[i]] for i in range(batch_size)], dim=0)
        return poly

    def __call__(self, batch):
        
        ct_num = torch.tensor([b['ct_num'] for b in batch])

        data_dict = {}
        max_len = torch.max(ct_num)
        batch_size = len(batch)
        ct_01 = torch.zeros([batch_size, max_len], dtype=torch.bool)
        ct_img_idx = torch.zeros([batch_size, max_len], dtype=torch.int64)
        for i in range(batch_size):
            ct_01[i, :ct_num[i]] = 1
            ct_img_idx[i, :ct_num[i]] = i
        num_points_per_poly = 128
        unclip_ratio_boxes = torch.zeros([batch_size, max_len, num_points_per_poly, 2], dtype=torch.float)
        center_boxes = torch.zeros([batch_size, max_len, 2], dtype=torch.float)
        text_gt_polys = torch.zeros([batch_size,max_len,num_points_per_poly,2], dtype=torch.float)
        
        if max_len != 0:
            for n,b in enumerate(batch):
                for j in range(ct_num[n]):
                    unclip_ratio_boxes[n][j] = torch.tensor(np.ascontiguousarray( b['unclip_ratio_boxes'][j]))
                    center_boxes[n][j] = torch.tensor(np.ascontiguousarray(b['center'][j]))
                    text_gt_polys[n][j]= torch.tensor(np.ascontiguousarray( b['text_gt_polys'][j]))
        text_gt_polys = self.collect_training(text_gt_polys,ct_01)
        unclip_ratio_boxes = self.collect_training(unclip_ratio_boxes,ct_01)
        center_boxes = self.collect_training(center_boxes,ct_01)

        data_dict.update({'ct_01': ct_01,'ct_img_idx':ct_img_idx,'unclip_ratio_boxes': unclip_ratio_boxes,'center_boxes': center_boxes,"text_gt_polys":text_gt_polys
                       })
        to_tensor_keys = []
        for n,sample in enumerate(batch):
            for k, v in sample.items():
                if k in ['shrinked','expand','unclip_ratio_boxes','text_gt_polys','center','upclip']:
                    continue
                if k not in data_dict:
                    data_dict[k] = []
                if isinstance(v, (np.ndarray, torch.Tensor, PIL.Image.Image)):
                    if k not in to_tensor_keys:
                        to_tensor_keys.append(k)
                if isinstance(v, (np.ndarray, list)):
                    data_dict[k].append(torch.from_numpy(v))
                else:
                    data_dict[k].append(v)
        for k in to_tensor_keys:
            data_dict[k] = torch.stack(data_dict[k], 0)
        
        return data_dict

def get_dataloader(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    if data_path == None:
        return None

    data_path = [x for x in data_path if x is not None]
    
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()

    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
    sampler = None
    if distributed:
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(_dataset)
        config['loader']['shuffle'] = False
        config['loader']['pin_memory'] = True
    loader = DataLoader(dataset=_dataset, sampler=sampler, **config['loader'])
    return loader

def get_dataset_(module_config, distributed=False):
    if module_config is None:
        return None
    config = copy.deepcopy(module_config)
    dataset_args = config['dataset']['args']
    if 'transforms' in dataset_args:
        img_transfroms = get_transforms(dataset_args.pop('transforms'))
    else:
        img_transfroms = None
    dataset_name = config['dataset']['type']
    data_path = dataset_args.pop('data_path')
    if data_path == None:
        return None

    data_path = [x for x in data_path if x is not None]
    
    if len(data_path) == 0:
        return None
    if 'collate_fn' not in config['loader'] or config['loader']['collate_fn'] is None or len(config['loader']['collate_fn']) == 0:
        config['loader']['collate_fn'] = None
    else:
        
        config['loader']['collate_fn'] = eval(config['loader']['collate_fn'])()
    _dataset = get_dataset(data_path=data_path, module_name=dataset_name, transform=img_transfroms, dataset_args=dataset_args)
  
    return _dataset