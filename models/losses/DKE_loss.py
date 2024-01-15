# -*- coding: utf-8 -*-

# from telnetlib import DM
from torch import nn
import torch
from models.losses.basic_loss import BalanceCrossEntropyLoss, MaskL1Loss, DiceLoss

    

class DMLoss(nn.Module):
    def __init__(self, type='smooth_l1',isnearest=True,isinit = False):
        type_list = {'l1': torch.nn.functional.l1_loss, 'smooth_l1': torch.nn.functional.smooth_l1_loss,'l2':torch.nn.functional.mse_loss}
        self.crit = type_list[type]
        # print('loss:',)
        # from scipy.optimize import linear_sum_assignment
        self.isnearest = isnearest
        self.isinit = isinit
        super(DMLoss, self).__init__()

    def interpolation(self, poly, time=5):
        ori_points_num = poly.size(1)
        poly_roll =torch.roll(poly, shifts=1, dims=1)
        poly_ = poly.unsqueeze(3).repeat(1, 1, 1, time)
        poly_roll = poly_roll.unsqueeze(3).repeat(1, 1, 1, time)
        step = torch.arange(0, time, dtype=torch.float32).cuda() / time
        poly_interpolation = poly_ * step + poly_roll * (1. - step)
        poly_interpolation = poly_interpolation.permute(0, 1, 3, 2).reshape(poly_interpolation.size(0), ori_points_num * time, 2)
        return poly_interpolation
    
    def compute_distance(self, pred_poly, gt_poly):
        pred_poly_expand = pred_poly.unsqueeze(1)
        gt_poly_expand = gt_poly.unsqueeze(2)
        gt_poly_expand = gt_poly_expand.expand(gt_poly_expand.size(0), gt_poly_expand.size(1),
                                               pred_poly_expand.size(2), gt_poly_expand.size(3))
        pred_poly_expand = pred_poly_expand.expand(pred_poly_expand.size(0), gt_poly_expand.size(1),
                                                   pred_poly_expand.size(2), pred_poly_expand.size(3))
        distance = torch.sum((pred_poly_expand - gt_poly_expand) ** 2, dim=3)
        return distance
    
    def lossPred2NearestGt(self,init_polys, pred_poly, gt_poly):
        if not self.isnearest:
            from scipy.optimize import linear_sum_assignment
            gt_poly_interpolation = self.interpolation(gt_poly,time=1)
            _gt_poly_interpolation = gt_poly_interpolation.clone().detach().cpu()
            if self.isinit:
                _init_polys = init_polys.clone().detach().cpu()
            else:
                _init_polys = pred_poly.clone().detach().cpu()
            distance_pred_gtInterpolation = self.compute_distance(_gt_poly_interpolation,_init_polys)
            col_list = []
            for n,cost in enumerate(distance_pred_gtInterpolation):
                row_ind, col_ind = linear_sum_assignment(cost)
                col_list.append(gt_poly_interpolation[n][col_ind].unsqueeze(0))
            gt_poly_interpolation = torch.cat(col_list,dim=0)
            loss_ini_pred_poly_nearestgt = self.crit(pred_poly,gt_poly_interpolation)
            return loss_ini_pred_poly_nearestgt
        if self.isinit:
            _init_polys = init_polys.clone().detach().cpu()
        else:
            _init_polys = pred_poly.clone().detach().cpu()
        gt_poly_interpolation = self.interpolation(gt_poly)
        n = len(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(_init_polys, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_ini_pred_poly_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_ini_pred_poly_nearestgt


    def loss2NearestGt(self, ini_pred_poly, pred_poly, gt_poly):
        gt_poly_interpolation = self.interpolation(gt_poly)
        distance_pred_gtInterpolation = self.compute_distance(ini_pred_poly, gt_poly_interpolation)
        index_gt = torch.min(distance_pred_gtInterpolation, dim=1)[1]
        index_0 = torch.arange(index_gt.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_gt.size(0), index_gt.size(1))
        loss_predto_nearestgt = self.crit(pred_poly,gt_poly_interpolation[index_0, index_gt, :])
        return loss_predto_nearestgt
    def lossGt2NearestPred(self, ini_pred_poly, pred_poly, gt_poly):
        distance_pred_gt = self.compute_distance(ini_pred_poly, gt_poly)
        index_pred = torch.min(distance_pred_gt, dim=2)[1]
        index_0 = torch.arange(index_pred.size(0))
        index_0 = index_0.unsqueeze(1).expand(index_pred.size(0), index_pred.size(1))
        loss_gtto_nearestpred = self.crit(pred_poly[index_0, index_pred, :], gt_poly,reduction='none')
        return loss_gtto_nearestpred

    def setloss(self, ini_pred_poly, pred_poly, gt_poly, keyPointsMask):
        keyPointsMask = keyPointsMask.unsqueeze(2).expand(keyPointsMask.size(0), keyPointsMask.size(1), 2)
        lossPred2NearestGt = self.lossPred2NearestGt(ini_pred_poly, pred_poly, gt_poly)
        lossGt2NearestPred = self.lossGt2NearestPred(ini_pred_poly, pred_poly, gt_poly)

        loss_set2set = torch.sum(lossGt2NearestPred * keyPointsMask) / (torch.sum(keyPointsMask) + 1) + lossPred2NearestGt
        return loss_set2set / 2.

    def setNearestloss(self, init_polys,ini_pred_poly, gt_poly):
        lossPred2NearestGt = self.lossPred2NearestGt(init_polys,ini_pred_poly, gt_poly)
        return lossPred2NearestGt

    def forward(self, init_polys,pred_poly, gt_polys, keyPointsMask=None):
        if keyPointsMask is None:
            return self.setNearestloss(init_polys,pred_poly, gt_polys)

      
class SEG_e2ec_dml(nn.Module):
    def __init__(self, alpha=1.0, beta=1,gama=0.5, ohem_ratio=3, reduction='mean',isnearest=True,isinit=False, eps=1e-6,dice=False,_type='smooth_l1'):
        """
        :param alpha: balance binary_map loss 
        :param beta: balance regression loss 
        :param ohem_ratio
        :param reduction
        """
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.dice = dice
        if self.dice:
            self.dice_loss = DiceLoss(eps=eps)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.dmlloss = DMLoss(type=_type,isnearest=isnearest,isinit=isinit)
  
    
    def forward(self, predict, batch):
        pred = predict['pred']
        coarse_polys = predict['coarse_polys']
        _key = predict.keys()
        init_polys = predict['init_polys']
        shrink_maps = pred[:, 0, :, :]
        if "pred_polys" in _key:
            pred_polys = predict['pred_polys']
        n = len(coarse_polys) 
        coarse_py_loss = self.dmlloss(init_polys,coarse_polys,batch['text_gt_polys'])
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps,coarse_py_loss=coarse_py_loss)
        if self.dice:
            dice_loss_maps = self.dice_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
            metrics.update({'dice_loss':dice_loss_maps})

        loss_all = self.alpha * loss_shrink_maps+self.beta * coarse_py_loss 
        if self.dice:
            loss_all = loss_all + self.gama*dice_loss_maps
        if "pred_polys" in _key:
            pred_py_loss = self.dmlloss(init_polys,pred_polys, batch['text_gt_polys'])
            metrics.update(pred_py_loss=pred_py_loss)
            loss_all = loss_all+pred_py_loss*self.gama
        metrics['loss'] = loss_all
        return metrics

class SEG_e2ec(nn.Module):
    def __init__(self, alpha=1.0, beta=10,gama=0.5, ohem_ratio=3, reduction='mean', eps=1e-6):
        super().__init__()
        assert reduction in ['mean', 'sum'], " reduction must in ['mean','sum']"
        self.alpha = alpha
        self.beta = beta
        self.gama = gama
        self.bce_loss = BalanceCrossEntropyLoss(negative_ratio=ohem_ratio)
        self.ohem_ratio = ohem_ratio
        self.reduction = reduction
        self.py_crit = torch.nn.functional.smooth_l1_loss
    
    def forward(self, predict, batch):
        _key = predict.keys()
        pred = predict['pred']
        coarse_polys = predict['coarse_polys']
        if "pred_polys" in _key:
            pred_polys = predict['pred_polys']
        shrink_maps = pred[:, 0, :, :]
        coarse_py_loss = self.py_crit(coarse_polys, batch['text_gt_polys'])
        loss_shrink_maps = self.bce_loss(shrink_maps, batch['shrink_map'], batch['shrink_mask'])
        metrics = dict(loss_shrink_maps=loss_shrink_maps,coarse_py_loss=coarse_py_loss)
        loss_all = self.alpha * loss_shrink_maps+self.beta*coarse_py_loss 
        if "pred_polys" in _key:
            pred_py_loss = self.py_crit(pred_polys, batch['text_gt_polys'])
            metrics.update(pred_py_loss=pred_py_loss)
            loss_all = loss_all+pred_py_loss*self.gama

        metrics['loss'] = loss_all
        return metrics