import cv2
import numpy as np
from.utils import  get_img_gt,multiLine2Points,calculate_polygon_area,_idx

class Makexpansion():
    def __init__(self, max_candidates=1000,num_points_per_poly=128):
        self.min_size = 3
        self.max_candidates = max_candidates
        self.num_points_per_poly = num_points_per_poly

    def __call__(self, data):
       
        height, width,_ = data['img'].shape
        ignore_tags = data['ignore_tags']
        self.ignore_tags = ignore_tags
        assert len(ignore_tags) == len(data['text_polys'])
        boxes,centers,unclip_ratio_boxes,text_gt_polys = self.polygons_from_data(data,width, height)
        self.shrinked = data['shrinked']
        data['center'] = np.array(centers)
        data['unclip_ratio_boxes'] =  unclip_ratio_boxes
        data['ct_num'] = len(text_gt_polys)
        data['text_gt_polys'] =   text_gt_polys
        assert len(centers) == len(text_gt_polys)
       
        return data
   
    def order_counters_clockwise(self,pts):
      
        cond = []
        for n,re in enumerate(pts):
            rect= np.zeros((4,2),dtype="float32")
            s = re.sum(axis=1)
            rect[0] = re[np.argmin(s)]
            rect[2] = re[np.argmax(s)]
            diff = np.diff(re, axis=1)
            rect[1] = re[np.argmin(diff)]
            rect[3] = re[np.argmax(diff)]
            cond.append(rect)
        return cond
 
    def polygons_from_data(self,data,width, height):
        '''
        _bitmap: single map with shape (H, W),
            whose values are binarized as {0, 1}
        '''
       
        dest_width = width
        dest_height = height
        contours = data['shrinked']
        upclip_contours  =data['upclip']
        assert len(contours)==len(upclip_contours)
        boxes = []
        centers = []
        refine_boxes = []
        text_gt_polys= []
        for n,contour in enumerate(contours[:self.max_candidates]):
            
            if  np.squeeze(contour).shape[0] <4:
                continue
            real_contour = upclip_contours[n]
            contour[:, 0] = np.clip(contour[:, 0], 0, width - 1)
            contour[:, 1] = np.clip(contour[:, 1], 0, height - 1)
            if not isinstance(dest_width, int):
                dest_width = dest_width.item()
                dest_height = dest_height.item()
            boxes.append(contour)
            contour = multiLine2Points(contour, 1).astype(np.int32)
            real_contour = multiLine2Points(real_contour, 1).astype(np.int32)
            cx,cy =0,0
            centers.append([int(cx),int(cy)])
            refine_boxes.append(self.return_counters(contour, height,width,int(cx),int(cy)))
            text_gt_polys.append(self.return_counters(real_contour,height,width,int(cx),int(cy)))

        return boxes,centers,refine_boxes,text_gt_polys


   
    def return_counters(self,contour,height,width,cx,cy):
        if calculate_polygon_area(contour)<0:
            contour = np.flipud(contour)
        idx = _idx(contour,cx,cy,2) #four_idx(contour,cx,cy)
        contour = get_img_gt(contour, idx,t=self.num_points_per_poly)
        return contour

