import torch
import cv2
import numpy as np
from .utils import get_img_gt,_idx
import torch.nn as nn
from .snake import Snake
from .utils import img_poly_to_can_poly_evo,get_gcn_feature_evo,get_gcn_feature_test_evo,multiLine2Points,calculate_polygon_area


class Evolution(nn.Module):
    def __init__(self, evole_ietr_num=0,num_point=128, evolve_stride=4., ro=1.,feature_dim=256):
        super(Evolution, self).__init__()
        self.evolve_stride = evolve_stride
        self.ro = ro
        self.num_point = num_point
        self.evolve_gcn = Snake(state_dim=num_point, feature_dim=feature_dim+2, conv_type='dgrid')
        self.evolve_dcn = Snake(state_dim=num_point, feature_dim=feature_dim+2, conv_type='dgrid')
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0.0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


    def prepare_testing_evolve(self, output, h, w):
        img_init_polys = output['img_init_polys']
        img_init_polys[..., 0] = torch.clamp(img_init_polys[..., 0], min=0, max=w-1)
        img_init_polys[..., 1] = torch.clamp(img_init_polys[..., 1], min=0, max=h-1)
        output.update({'img_init_polys': img_init_polys})
        return img_init_polys
    
    def evolve_poly(self, snake, cnn_feature, batch,i_it_poly, c_it_poly, ignore=False):
        
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        poly_num =len(i_it_poly)
        if poly_num!=0:
            init_feature = get_gcn_feature_evo(cnn_feature,i_it_poly,batch,self.evolve_stride).view(poly_num,cnn_feature.size(1),-1)
            
            c_it_poly = c_it_poly * self.ro
            init_input = torch.cat([init_feature, c_it_poly.permute(0, 2, 1)], dim=1)
            offset = snake(init_input).permute(0, 2, 1)
            i_poly = i_it_poly * self.ro + offset * self.evolve_stride
        else:
            return i_it_poly
        return i_poly


    def evolve_poly_test(self, snake, cnn_feature,i_it_poly, c_it_poly):
        # expanding or fine-tuning 
        h, w = cnn_feature.size(2), cnn_feature.size(3)
        poly_num =len(i_it_poly)
        # get vertex features
        init_feature =get_gcn_feature_test_evo(cnn_feature=cnn_feature, init_polys=i_it_poly).contiguous().view(poly_num,cnn_feature.size(1),-1)
        c_it_poly = (c_it_poly * self.ro).to(cnn_feature.device)
        init_input = torch.cat([init_feature, c_it_poly.permute(0,2,1)], dim=1)
        init_input = init_input.type(torch.float32).cuda()
        # get offsets
        offset = snake(init_input).permute(0, 2, 1).cpu()
        i_it_poly = i_it_poly.clone().detach().cpu()
        i_poly = i_it_poly * self.ro + offset * self.evolve_stride
        return i_poly

    def foward_train(self, batch, cnn_feature,ignore=False):
        result = {}
        init_polys = batch['unclip_ratio_boxes'] / self.evolve_stride
        coarse_polys = self.evolve_poly(self.evolve_gcn, cnn_feature,batch,i_it_poly = init_polys,
                                 c_it_poly= img_poly_to_can_poly_evo(init_polys))
        result.update({'init_polys':init_polys,'coarse_polys':coarse_polys*self.evolve_stride})
        if ignore:
            return result
        pred_polys = self.evolve_poly(self.evolve_dcn, cnn_feature,batch,i_it_poly = coarse_polys,
                                 c_it_poly= img_poly_to_can_poly_evo(coarse_polys))
        result.update({'init_polys':init_polys,'pred_polys':pred_polys*self.evolve_stride})
        return result
    
    

    def foward_test(self, cnn_feature,init_polys,ignore):
        result = {}
        
        if len(init_polys)==0:
            if ignore:
                result.update({'coarse_polys':torch.tensor([])})
            else:
                result.update({'coarse_polys':torch.tensor([]),'pred_polys':torch.tensor([])})
            return result
        init_polys = torch.tensor(init_polys)
        init_polys = init_polys/self.evolve_stride
        result = {}
        with torch.no_grad():
            coarse_polys = self.evolve_poly_test(self.evolve_gcn, cnn_feature=cnn_feature,i_it_poly = init_polys,
                                 c_it_poly= img_poly_to_can_poly_evo(init_polys))
            result.update({'coarse_polys':coarse_polys*self.evolve_stride})
            if ignore:
                return result
            pred_polys = self.evolve_poly_test(self.evolve_dcn, cnn_feature,i_it_poly = coarse_polys,
                                 c_it_poly= img_poly_to_can_poly_evo(coarse_polys))
            result.update({'pred_polys':pred_polys*self.evolve_stride})
        return result

    def forward(self,cnn_feature, batch,is_training=True):
        return self.foward_train(batch,cnn_feature)
 

class Decode(torch.nn.Module):
    # c_in should be 64
    def __init__(self, c_in=256, num_point=128,ro=1.0, evolve_stride=4., down_sample=4., thresh=0.3,box_thresh=0.7,epsilon=0.002,min_size=4,is_output_polygon=True,ignore=False):

        super(Decode, self).__init__()
        self.min_size = min_size
        self.down_sample = down_sample
        self.box_thresh = box_thresh
        self.thresh = thresh
        self.num_point = num_point
        self.epsilon= epsilon
        self.ignore = ignore
        self.is_output_polygon = is_output_polygon
        self.gcn = Evolution(evolve_stride=evolve_stride,num_point=num_point,
                            ro=ro,feature_dim=c_in)

    def train_decode(self, batch, cnn_feature):
        result = self.gcn.foward_train(batch,cnn_feature,self.ignore)
        return result
    

    def test_decode(self, batch,  cnn_feature):
        cnn_feature = cnn_feature.cpu()
        pred = batch['pred'][:, 0, :, :]
        pred = pred.cpu()
        segmentation = self.binarize(pred,self.thresh)
        for batch_index in range(pred.size(0)):
            dest_height, dest_width = batch['shape'][batch_index]
            boxes,centers, scores = self.polygons_from_bitmap(pred[batch_index], segmentation[batch_index], dest_width, dest_height)
            boxes_batch = boxes
            scores_batch= scores
        
        result = self.gcn.foward_test(cnn_feature,boxes_batch,self.ignore)

        coarse_polys = result['coarse_polys']

        # after plenty of experiments, we decide to abandon fine-tuning procedure(iter.2)
        if "pred_polys" in result.keys():
            pred_polys =  result['pred_polys']

        if coarse_polys.shape[0] == 0:
            return result
        
        height, width = segmentation[0].shape

        coarse_polys[:,:,0] = np.clip(np.round(coarse_polys[:,:, 0] / width * dest_width), 0, dest_width)
        coarse_polys[:,:,1] = np.clip(np.round(coarse_polys[:,:, 1] / height * dest_height), 0, dest_height)
        result.update({'coarse_polys': self.modify(coarse_polys),'scores':scores})
        if self.ignore:
            return result
        pred_polys[:,:,0] = np.clip(np.round(pred_polys[:,:, 0] / width * dest_width), 0, dest_width)
        pred_polys[:,:,1] = np.clip(np.round(pred_polys[:,:, 1] / height * dest_height), 0, dest_height)
        result.update({'pred_polys': self.modify(pred_polys),'scores':scores})
        return result
    def order_points_clockwise(self,pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    def modify(self,contours):
        contour_ = []
        if self.epsilon==0 and self.is_output_polygon:
            return contours
        for contour in contours:
            contour = np.array(contour, dtype=np.int32)
            if self.is_output_polygon:
                epsilon = self.epsilon * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = np.flipud(approx.reshape((-1, 2)))
                contour_.append(points)
            else:
                box, sside = self.get_mini_boxes(contour)
                if sside < self.min_size:
                    continue
                box = np.array(box)
                contour_.append(self.order_points_clockwise(box))
        return [np.array(contour_)]  

    def forward(self, data_input, cnn_feature, is_training=True, ignore_gloabal_deform=False):
        if is_training:
            return self.train_decode(data_input,  cnn_feature)
        else:
            return self.test_decode(data_input,  cnn_feature)
        
    def binarize(self, pred,thresh):
        return pred > thresh
    def get_mini_boxes(self, contour):

        bounding_box = cv2.minAreaRect(contour)
        points = sorted(list(cv2.boxPoints(bounding_box)), key=lambda x: x[0])

        index_1, index_2, index_3, index_4 = 0, 1, 2, 3
        if points[1][1] > points[0][1]:
            index_1 = 0
            index_4 = 1
        else:
            index_1 = 1
            index_4 = 0
        if points[3][1] > points[2][1]:
            index_2 = 2
            index_3 = 3
        else:
            index_2 = 3
            index_3 = 2

        box = [points[index_1], points[index_2], points[index_3], points[index_4]]
        return box, min(bounding_box[1])
 
    def polygons_from_bitmap(self, pred, _bitmap, dest_width, dest_height):
        '''
        _bitmap: single map with shape (H, W),
        pred: whose values are binarized as {0, 1}
        '''
        assert len(_bitmap.shape) == 2
        # pred = pred.cpu()
        bitmap = _bitmap.numpy()  
        pred = pred.detach().numpy()
        height, width = bitmap.shape
        boxes = []
        scores = []
        centers = []
        contours, _ = cv2.findContours((bitmap * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        for contour in contours:
            points = contour.reshape((-1, 2))
            if points.shape[0] < 4:
                continue
            _, sside = self.get_mini_boxes(contour)
            if sside<self.min_size:
                continue
            score = self.box_score_fast(pred, contour.squeeze(1))
            if self.box_thresh > score:
                continue
            contour = np.squeeze(contour)
            if calculate_polygon_area(contour)<0:
                contour = np.flipud(contour)
            boxes.append(self.return_counters(contour))
            scores.append(score)
        return boxes,[], scores
    
    def return_counters(self,contour,cx=0,cy=0):
        idx = _idx(contour,cx,cy,2)
        contour = get_img_gt(contour, idx)
        return contour
    
    def box_score_fast(self, bitmap, _box):
        h, w = bitmap.shape[:2]
        box = _box.copy()
        xmin = np.clip(np.floor(box[:, 0].min()).astype(np.int), 0, w - 1)
        xmax = np.clip(np.ceil(box[:, 0].max()).astype(np.int), 0, w - 1)
        ymin = np.clip(np.floor(box[:, 1].min()).astype(np.int), 0, h - 1)
        ymax = np.clip(np.ceil(box[:, 1].max()).astype(np.int), 0, h - 1)

        mask = np.zeros((ymax - ymin + 1, xmax - xmin + 1), dtype=np.uint8)
        box[:, 0] = box[:, 0] - xmin
        box[:, 1] = box[:, 1] - ymin
        cv2.fillPoly(mask, box.reshape(1, -1, 2).astype(np.int32), 1)
        return cv2.mean(bitmap[ymin:ymax + 1, xmin:xmax + 1], mask)[0]

