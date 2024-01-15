from addict import Dict
from torch import nn
import torch.nn.functional as F
import time
import torch
from models.backbone import build_backbone
from models.neck import build_neck
from models.head import build_head
from .decode import build_decode

class DKE(nn.Module):
    def __init__(self, model_config: dict):
        """
        :param model_config: for initializing the model
        """
        super().__init__()
        model_config = Dict(model_config)
        backbone_type = model_config.backbone.pop('type')
        neck_type = model_config.neck.pop('type') 
        head_type = model_config.head.pop('type')
        decode_type = model_config.decode.pop('type')
        self.backbone = build_backbone(backbone_type, **model_config.backbone)
        self.neck = build_neck(neck_type, in_channels=self.backbone.out_channels, **model_config.neck)
        self.head = build_head(head_type, in_channels=self.neck.out_channels, **model_config.head)
        self.name = f'{backbone_type}_{neck_type}_{head_type}'
        self.decode = build_decode(decode_type, **model_config.decode)
    def forward(self, x,batch,is_training = True):
        _, _, H, W = x.size()
        backbone_out = self.backbone(x)
        neck_out = self.neck(backbone_out)
        y = self.head(neck_out)
        y = F.interpolate(y, size=(H, W), mode='bilinear', align_corners=True)
        if not is_training:
            # y is the probability for visualization and contour extraction
            batch['pred'] = y 
            result = self.decode(batch,neck_out,is_training=is_training)
            batch['pred'] = batch['pred'].detach().cpu()
            result.update({'pred':batch['pred']})
            return result
        else:
            batch['pred'] = y
            result = self.decode(batch,neck_out,is_training=is_training)
            result.update({'pred':batch['pred']})
            return result
 