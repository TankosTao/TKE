# -*- coding: utf-8 -*-
import time

import torch
import torchvision.utils as vutils
from tqdm import tqdm

from base import BaseTrainer
from utils import WarmupPolyLR, runningScore, cal_text_score
from utils.util import save_result


class Trainer(BaseTrainer):
    def __init__(self, config, model, criterion, train_loader, validate_loader, metric_cls, post_process=None):
        super(Trainer, self).__init__(config, model, criterion)
        self.train_loader = train_loader
        if validate_loader is not None:
            assert metric_cls is not None
        self.validate_loader = validate_loader
        self.post_process = post_process
        self.metric_cls = metric_cls
        self.train_loader_len = len(train_loader)
        self.save_checkpoint_while_log = self.config['trainer']['save_checkpoint_while_log']
        if isinstance(config['lr_scheduler'], dict):
            if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                warmup_iters = config['lr_scheduler']['args']['warmup_epoch'] * self.train_loader_len
                if self.start_epoch > 1:
                    self.config['lr_scheduler']['args']['last_epoch'] = (self.start_epoch - 1) * self.train_loader_len
                self.scheduler = WarmupPolyLR(self.optimizer, max_iters=self.epochs * self.train_loader_len,
                                            warmup_iters=warmup_iters, **config['lr_scheduler']['args'])
        self.save_model_intr = self.config['trainer']['save_model_intr']
        if self.validate_loader is not None:
            self.eval_epoch = self.config['trainer']['eval_every_epoch']
            self.logger_info(
                'train dataset has {} samples,{} in dataloader, validate dataset has {} samples,{} in dataloader'.format(
                    len(self.train_loader.dataset), self.train_loader_len, len(self.validate_loader.dataset), len(self.validate_loader)))
        else:
            self.logger_info('train dataset has {} samples,{} in dataloader'.format(len(self.train_loader.dataset), self.train_loader_len))

    def _train_epoch(self, epoch):
        self.model.train()
        epoch_start = time.time()
        batch_start = time.time()
        train_loss = 0.
        running_metric_text = runningScore(2)
        lr = self.optimizer.param_groups[0]['lr']
        for i, batch in enumerate(self.train_loader):
         
            if self.global_step!=0 and self.global_step%self.save_model_intr==0:
                net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
                self._save_checkpoint(epoch, net_save_path)
                self.logger_info('save checkpoint to {}'.format(net_save_path))
            if self.global_step!=0 and self.global_step%50000 == 0:
                net_save_path = '{}/model_{}.pth'.format(self.checkpoint_dir,self.global_step)
                self._save_checkpoint(epoch, net_save_path)
                self.logger_info('save checkpoint to {}'.format(net_save_path))
            if i >= self.train_loader_len:
                break
            self.global_step += 1
            if isinstance(self.config['lr_scheduler'], str):
                self.adjust_learning_rate(self.optimizer, self.train_loader, epoch, i, self.config)
            
            lr = self.optimizer.param_groups[0]['lr']

            for key, value in batch.items():
                if value is not None:
                    if isinstance(value, torch.Tensor):
                        batch[key] = value.to(self.device)
            cur_batch_size = batch['img'].size()[0]
            preds = self.model(batch['img'],batch)
            loss_dict = self.criterion(preds, batch)
            self.optimizer.zero_grad()
            loss_dict['loss'].backward()
            self.optimizer.step()
            if isinstance(self.config['lr_scheduler'], dict):
                if self.config['lr_scheduler']['type'] == 'WarmupPolyLR':
                    self.scheduler.step()
            loss_str = 'loss: {:.4f}, '.format(loss_dict['loss'].item())
            for idx, (key, value) in enumerate(loss_dict.items()):
                loss_dict[key] = value.item()
                if key == 'loss':
                    continue
                loss_str += '{}: {:.4f}'.format(key, loss_dict[key])
                if idx < len(loss_dict) - 1:
                    loss_str += ', '

            train_loss += loss_dict['loss']
            if self.global_step % self.log_iter == 0:
                batch_time = time.time() - batch_start
                if self.wandb_enable:
                    
                    self.wandb.log({
                        'epoch':epoch,
                        'global_step':self.global_step,
                        'lr': '{:.6}'.format(lr),
                        **loss_dict,

                    })
                self.logger_info(
                    '[{}/{}], [{}/{}], global_step: {}, speed: {:.1f} samples/sec, {}, lr:{:.6}, time:{:.2f}'.format(
                        epoch, self.epochs, i + 1, self.train_loader_len, self.global_step, self.log_iter * cur_batch_size / batch_time, loss_str, lr, batch_time))
                batch_start = time.time()

        return {'train_loss': train_loss / self.train_loader_len, 'lr': lr, 'time': time.time() - epoch_start,
                'epoch': epoch}

    def _eval(self, epoch):
        self.model.eval()
        # torch.cuda.empty_cache()  # speed up evaluating after training finished
        measure = {}
        raw_metrics = []
        raw_metrics_pred = []
        total_frame = 0.0
        total_time = 0.0
        for i, batch in tqdm(enumerate(self.validate_loader), total=len(self.validate_loader), desc='test model'):
            with torch.no_grad():
                batch['img'] = batch['img'].to(self.device)
                start = time.time()
                output = self.model(batch['img'],batch,is_training = False)
                boxes = output['coarse_polys']
                if 'pred_polys' in output.keys():
                    boxes_pred_polys = output['pred_polys']
                    raw_metrics_pred.append(self.metric_cls.validate_measure(batch, boxes_pred_polys))

                total_frame += batch['img'].size()[0]
                total_time += time.time() - start
                raw_metric = self.metric_cls.validate_measure(batch, boxes)
                raw_metrics.append(raw_metric)
        if 'pred_polys' in output.keys():
            metrics_pred = self.metric_cls.gather_measure(raw_metrics_pred)
            measure.update({'pred':(metrics_pred['recall'].avg, metrics_pred['precision'].avg, metrics_pred['fmeasure'].avg)})
        metrics = self.metric_cls.gather_measure(raw_metrics)
        self.logger_info('FPS:{}'.format(total_frame / total_time))
        measure.update({'coarse':(metrics['recall'].avg, metrics['precision'].avg, metrics['fmeasure'].avg)})
        return measure

    def _on_epoch_finish(self):
        self.logger_info('[{}/{}], train_loss: {:.4f}, time: {:.4f}, lr: {}'.format(
            self.epoch_result['epoch'], self.epochs, self.epoch_result['train_loss'], self.epoch_result['time'],
            self.epoch_result['lr']))

        # save last
        net_save_path = '{}/model_final.pth'.format(self.checkpoint_dir)
        if self.epoch_result['epoch'] ==  self.config['trainer']['epochs']:
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
            net_save_path = '{}/model_latest.pth'.format(self.checkpoint_dir)
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
        save_best = False
        if self.validate_loader is not None and self.metric_cls is not None:  
            
            if self.epoch_result['epoch']%self.eval_epoch==0:
                measure = self._eval(self.epoch_result['epoch'])
                if 'coarse' in measure:
                    recall, precision, hmean = measure['coarse']
                    self.logger_info('test {}: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format('coarse',recall, precision, hmean))
                    if self.wandb_enable:
                        self.wandb.log({
                            'epoch':self.epoch_result['epoch'],
                            'coarse_recall':recall,
                            'coarse_precision': precision,
                            'coarse_f1':hmean
                        })

                if 'pred' in measure:
                    recall, precision, hmean = measure['pred']
                    if self.wandb_enable:
                        self.wandb.log({
                            # 'title': 'pred',
                            'epoch':self.epoch_result['epoch'],
                            'pred_recall':recall,
                            'pred_precision': precision,
                            'pred_f1':hmean
                        })
                    self.logger_info('test {}: recall: {:.6f}, precision: {:.6f}, f1: {:.6f}'.format('pred',recall, precision, hmean))
                if self.tensorboard_enable:
                    self.writer.add_scalar('EVAL/recall', recall, self.global_step)
                    self.writer.add_scalar('EVAL/precision', precision, self.global_step)
                    self.writer.add_scalar('EVAL/hmean', hmean, self.global_step)
                

                if hmean >= self.metrics['hmean'] :
                    save_best = True
                    self.metrics['train_loss'] = self.epoch_result['train_loss']
                    self.metrics['hmean'] = hmean
                    self.metrics['precision'] = precision
                    self.metrics['recall'] = recall
                    self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        else:
            if self.epoch_result['train_loss'] <= self.metrics['train_loss']:
                save_best = True
                self.metrics['train_loss'] = self.epoch_result['train_loss']
                self.metrics['best_model_epoch'] = self.epoch_result['epoch']
        best_str = 'current best, '
        for k, v in self.metrics.items():
            best_str += '{}: {:.6f}, '.format(k, v)
        self.logger_info(best_str)
        if self.validate_loader is not None and save_best and self.epoch_result['epoch']%self.eval_epoch==0 :
            net_save_path_best = '{}/model_best.pth'.format(self.checkpoint_dir)
            self._save_checkpoint(self.epoch_result['epoch'], net_save_path_best)
            self.logger_info("Saving current best: {}".format(net_save_path_best))

    def _on_train_finish(self):
        for k, v in self.metrics.items():
            self.logger_info('{}:{}'.format(k, v))

        net_save_path = '{}/model_final.pth'.format(self.checkpoint_dir)
        self._save_checkpoint(self.epoch_result['epoch'], net_save_path)
        self.logger_info('save checkpoint to {}'.format(net_save_path))
        self.logger_info('finish train')
    
    def adjust_learning_rate(self,optimizer, dataloader, epoch, iter, config):
        schedule = config['lr_scheduler']
        if isinstance(schedule, str):
            if schedule == 'polylr':
                cur_iter = (epoch-1) * len(dataloader) + iter
                max_iter_num = config['trainer']['epochs'] * len(dataloader)
                lr = config['optimizer']['args']['lr'] * (1 - float(cur_iter) / max_iter_num) ** 0.9
            elif schedule == 'fixlr':
                lr = config['optimizer']['args']['lr']
            else:
                raise ValueError('Schedule should be polylr or fixlr!')
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
        

