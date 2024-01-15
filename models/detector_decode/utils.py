from cmath import e
from distutils.command.config import config
import numpy as np
import cv2
import random
from shapely.geometry import Polygon
import torch

def get_adj_mat(n_adj, n_nodes):
    a = np.zeros([n_nodes, n_nodes], dtype=np.float)

    for i in range(n_nodes):
        for j in range(-n_adj // 2, n_adj // 2 + 1):
            if j != 0:
                a[i][(i + j) % n_nodes] = 1
                a[(i + j) % n_nodes][i] = 1
    return a


def get_3rd_point(a, b):
    direct = a - b
    return b + np.array([-direct[1], direct[0]], dtype=np.float32)


def get_dir(src_point, rot_rad):
    sn, cs = np.sin(rot_rad), np.cos(rot_rad)

    src_result = [0, 0]
    src_result[0] = src_point[0] * cs - src_point[1] * sn
    src_result[1] = src_point[0] * sn + src_point[1] * cs

    return src_result

def get_affine_transform(center,
                         scale,
                         rot,
                         output_size,
                         shift=np.array([0, 0], dtype=np.float32),
                         inv=0):
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)

    scale_tmp = scale
    src_w = scale_tmp[0]
    dst_w = output_size[0]
    dst_h = output_size[1]

    rot_rad = np.pi * rot / 180
    src_dir = get_dir([0, src_w * -0.5], rot_rad)
    dst_dir = np.array([0, dst_w * -0.5], np.float32)

    src = np.zeros((3, 2), dtype=np.float32)
    dst = np.zeros((3, 2), dtype=np.float32)
    src[0, :] = center + scale_tmp * shift
    src[1, :] = center + src_dir + scale_tmp * shift
    dst[0, :] = [dst_w * 0.5, dst_h * 0.5]
    dst[1, :] = np.array([dst_w * 0.5, dst_h * 0.5], np.float32) + dst_dir

    src[2:, :] = get_3rd_point(src[0, :], src[1, :])
    dst[2:, :] = get_3rd_point(dst[0, :], dst[1, :])
    # print('src,dst',src,dst)
    if inv:
        trans = cv2.getAffineTransform(np.float32(dst), np.float32(src))
    else:
        trans = cv2.getAffineTransform(np.float32(src), np.float32(dst))
    return trans


def affine_transform(pt, t):
    """pt: [n, 2]"""
    new_pt = np.dot(np.array(pt), t[:, :2].T) + t[:, 2]
    return new_pt


def get_border(border, size):
    i = 1
    while np.any(size - border // i <= border // i):
        i *= 2
    return border // i

def grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def lighting_(data_rng, image, alphastd, eigval, eigvec):
    alpha = data_rng.normal(scale=alphastd, size=(3, ))
    image += np.dot(eigvec, eigval * alpha)

def blend_(alpha, image1, image2):
    image1 *= alpha
    image2 *= (1 - alpha)
    image1 += image2

def saturation_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs[:, :, None])

def brightness_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    image *= alpha

def contrast_(data_rng, image, gs, gs_mean, var):
    alpha = 1. + data_rng.uniform(low=-var, high=var)
    blend_(alpha, image, gs_mean)

def color_aug(data_rng, image, eig_val, eig_vec):
    functions = [brightness_, contrast_, saturation_]
    random.shuffle(functions)
    gs = grayscale(image)
    gs_mean = gs.mean()
    for f in functions:
        f(data_rng, image, gs, gs_mean, 0.4)
    lighting_(data_rng, image, 0.1, eig_val, eig_vec)

def augment(img, split, cfg):
    _data_rng, _eig_val, _eig_vec, mean, std,down_ratio, input_h, input_w, scale_range, scale, test_rescale, test_scale= cfg.data.data_rng,cfg.data.eig_val, cfg.data.eig_vec, cfg.data.mean, cfg.data.std, cfg.commen.down_ratio,cfg.data.input_h, cfg.data.input_w,cfg.data.scale_range,cfg.data.scale, cfg.test.test_rescale, cfg.data.test_scale

    
    
    height, width = img.shape[0], img.shape[1]
    center = np.array([img.shape[1] / 2., img.shape[0] / 2.], dtype=np.float32)
    if scale is None:
        scale = max(img.shape[0], img.shape[1]) * 1.0
    if not isinstance(scale, np.ndarray) and not isinstance(scale, list):
        scale = np.array([scale, scale], dtype=np.float32)


    flipped = False
    if split == 'train':
        scale = scale
        x, y = center
        w_border = get_border(width/4, scale[0]) + 1 
        # 160+1
        h_border = get_border(height/4, scale[0]) + 1
        center[0] = np.random.randint(low=max(x - w_border, 0), high=min(x + w_border, width - 1))
        center[1] = np.random.randint(low=max(y - h_border, 0), high=min(y + h_border, height - 1))

    if split != 'train':
        scale = np.array([width, height])
        x = 32
        if test_rescale is not None:
            # print(test_rescale)
            _scale = (test_rescale/np.min(np.array([width, height])))*np.array([width, height])
            input_w, input_h = (int(_scale[0] / 1.) | (x - 1)) + 1,\
                               (int(_scale[1] / 1.) | (x - 1)) + 1
        else:
            # print(test_scale)
            if test_scale is None:
                input_w = (int(width / 1.) | (x - 1)) + 1
                input_h = (int(height / 1.) | (x - 1)) + 1
                # input_w,input_h = input_w*2,input_h*2
            else:
                scale = max(width, height) * 1.0
                scale = np.array([scale, scale])
                input_w, input_h = test_scale
        # print(input_w, input_h)
        center = np.array([width // 2, height // 2])
    inp = img.copy()
    if split != 'train':
        trans_input = get_affine_transform(center, scale, 0, [input_w, input_h])
        inp = cv2.warpAffine(img, trans_input, (input_w, input_h), flags=cv2.INTER_LINEAR)
    orig_img = inp.copy()

    inp = (inp.astype(np.float32) / 255.)

    inp = (inp - mean) / std
    inp = inp.transpose(2, 0, 1)
    output_h, output_w = input_h // down_ratio, input_w // down_ratio
    trans_output = get_affine_transform(center, scale, 0, [output_w, output_h])
    inp_out_hw = (input_h, input_w, output_h, output_w)
    trans_input = ''
    return orig_img, inp, trans_input, trans_output, flipped, center, scale, inp_out_hw

def handle_break_point(poly, axis, number, outside_border):
    if len(poly) == 0:
        return []

    if len(poly[outside_border(poly[:, axis], number)]) == len(poly):
        return []

    break_points = np.argwhere(
        outside_border(poly[:-1, axis], number) != outside_border(poly[1:, axis], number)).ravel()
    if len(break_points) == 0:
        return poly

    new_poly = []
    if not outside_border(poly[break_points[0], axis], number):
        new_poly.append(poly[:break_points[0]])

    for i in range(len(break_points)):
        current_poly = poly[break_points[i]]
        next_poly = poly[break_points[i] + 1]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])

        if outside_border(poly[break_points[i], axis], number):
            if mid_poly[axis] != next_poly[axis]:
                new_poly.append([mid_poly])
            next_point = len(poly) if i == (len(break_points) - 1) else break_points[i + 1]
            new_poly.append(poly[break_points[i] + 1:next_point])
        else:
            new_poly.append([poly[break_points[i]]])
            if mid_poly[axis] != current_poly[axis]:
                new_poly.append([mid_poly])

    if outside_border(poly[-1, axis], number) != outside_border(poly[0, axis], number):
        current_poly = poly[-1]
        next_poly = poly[0]
        mid_poly = current_poly + (next_poly - current_poly) * (number - current_poly[axis]) / (next_poly[axis] - current_poly[axis])
        new_poly.append([mid_poly])

    return np.concatenate(new_poly)

def transform_polys(polys, trans_output, output_h, output_w):
    new_polys = []
    for i in range(len(polys)):
        poly = polys[i]
        poly = affine_transform(poly, trans_output)
        poly = handle_break_point(poly, 0, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 0, output_w, lambda x, y: x >= y)
        poly = handle_break_point(poly, 1, 0, lambda x, y: x < y)
        poly = handle_break_point(poly, 1, output_h, lambda x, y: x >= y)
        if len(poly) == 0:
            continue
        if len(np.unique(poly, axis=0)) <= 2:
            continue
        new_polys.append(poly)
    return new_polys

def filter_tiny_polys(polys):
    return [poly for poly in polys if Polygon(poly).area > 5]

def get_cw_polys(polys):
    return [poly[::-1] if Polygon(poly).exterior.is_ccw else poly for poly in polys]

def gaussian_radius(det_size, min_overlap=0.7):
    height, width = det_size

    a1 = 1
    b1 = (height + width)
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    if b3 ** 2 - 4 * a3 * c3 < 0:
        r3 = min(r1, r2)
    else:
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)

def gaussian2D(shape, sigma=(1, 1), rho=0):
    if not isinstance(sigma, tuple):
        sigma = (sigma, sigma)
    sigma_x, sigma_y = sigma

    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m+1, -n:n+1]

    energy = (x * x) / (sigma_x * sigma_x) - 2 * rho * x * y / (sigma_x * sigma_y) + (y * y) / (sigma_y * sigma_y)
    h = np.exp(-energy / (2 * (1 - rho * rho)))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]
    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)
    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius + bottom, radius - left:radius + right]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0: # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap

def uniformsample(pgtnp_px2, newpnum):
    pnum, cnum = pgtnp_px2.shape
    assert cnum == 2

    idxnext_p = (np.arange(pnum, dtype=np.int32) + 1) % pnum
    pgtnext_px2 = pgtnp_px2[idxnext_p]
    edgelen_p = np.sqrt(np.sum((pgtnext_px2 - pgtnp_px2) ** 2, axis=1))
    edgeidxsort_p = np.argsort(edgelen_p)

    # two cases
    # we need to remove gt points
    # we simply remove shortest paths
    if pnum > newpnum:
        edgeidxkeep_k = edgeidxsort_p[pnum - newpnum:]
        edgeidxsort_k = np.sort(edgeidxkeep_k)
        pgtnp_kx2 = pgtnp_px2[edgeidxsort_k]
        assert pgtnp_kx2.shape[0] == newpnum
        return pgtnp_kx2
    # we need to add gt points
    # we simply add it uniformly
    else:
        edgenum = np.round(edgelen_p * newpnum / np.sum(edgelen_p)).astype(np.int32)
        for i in range(pnum):
            if edgenum[i] == 0:
                edgenum[i] = 1

        # after round, it may has 1 or 2 mismatch
        edgenumsum = np.sum(edgenum)
        if edgenumsum != newpnum:

            if edgenumsum > newpnum:

                id = -1
                passnum = edgenumsum - newpnum
                while passnum > 0:
                    edgeid = edgeidxsort_p[id]
                    if edgenum[edgeid] > passnum:
                        edgenum[edgeid] -= passnum
                        passnum -= passnum
                    else:
                        passnum -= edgenum[edgeid] - 1
                        edgenum[edgeid] -= edgenum[edgeid] - 1
                        id -= 1
            else:
                id = -1
                edgeid = edgeidxsort_p[id]
                edgenum[edgeid] += newpnum - edgenumsum

        assert np.sum(edgenum) == newpnum

        psample = []
        for i in range(pnum):
            pb_1x2 = pgtnp_px2[i:i + 1]
            pe_1x2 = pgtnext_px2[i:i + 1]

            pnewnum = edgenum[i]
            wnp_kx1 = np.arange(edgenum[i], dtype=np.float32).reshape(-1, 1) / edgenum[i]

            pmids = pb_1x2 * (1 - wnp_kx1) + pe_1x2 * wnp_kx1
            psample.append(pmids)

        psamplenp = np.concatenate(psample, axis=0)
        return psamplenp

def four_idx(img_gt_poly,cx,cy):
    can_gt_polys = img_gt_poly.copy()
    can_gt_polys[:, 0] -= cx
    can_gt_polys[:, 1] -= cy
    distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
    can_gt_polys = can_gt_polys.astype(np.float32)
    can_gt_polys /= np.repeat(distance, axis=1, repeats=2)
    idx_top = np.argmin(can_gt_polys[:, 1])
    return [idx_top]
def _idx(img_gt_poly,cx,cy,_type=1):
    if _type==1:
        return four_idx(img_gt_poly,cx,cy)
    if _type==2:
        can_gt_polys = img_gt_poly.astype(np.float32)
        x_min, y_min = np.min(img_gt_poly, axis=0)
        can_gt_polys[:, 0] -= x_min
        can_gt_polys[:, 1] -= y_min
        distance = np.sum(can_gt_polys ** 2, axis=1, keepdims=True) ** 0.5 + 1e-6
        idx_top = np.argmin(distance.squeeze())
    return [idx_top]
def get_img_gt(img_gt_poly, idx, t=128):
    align = len(idx)
    pointsNum = img_gt_poly.shape[0]
    r = []
    k = np.arange(0, t / align, dtype=float) / (t / align)
    for i in range(align):
        # print('begin')
        begin = idx[i]
        end = idx[(i + 1) % align]
        if align == 1:
            end = begin -1
        if begin > end:
            end += pointsNum
        r.append((np.round(((end - begin) * k).astype(int)) + begin) % pointsNum)
    r = np.concatenate(r, axis=0)
    return img_gt_poly[r, :]

def img_poly_to_can_poly(img_poly):
    x_min, y_min = np.min(img_poly, axis=0)
    can_poly = img_poly - np.array([x_min, y_min])
    return can_poly

def img_poly_to_can_poly_evo(img_poly):
    
    if len(img_poly) == 0:
        return torch.zeros_like(img_poly)
    # print(img_poly,np.array(img_poly).shape)
    x_min = torch.min(img_poly[..., 0], dim=-1)[0]
    y_min = torch.min(img_poly[..., 1], dim=-1)[0]
    can_poly = img_poly.clone()
    can_poly[..., 0] = can_poly[..., 0] - x_min[..., None]
    can_poly[..., 1] = can_poly[..., 1] - y_min[..., None]
    return can_poly

def get_gcn_feature_evo(cnn_feature,init_polys,batch,down_sample):
    ind = batch['ct_img_idx'][batch['ct_01'].bool()]
    batch_size,c,height, width = cnn_feature.shape
    img_poly = init_polys.clone()
    img_poly[..., 0] = (img_poly[..., 0]  )/(width/2.0)  -  1
    img_poly[..., 1] = (img_poly[..., 1]   )/(height/2.0)  - 1 

    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(cnn_feature.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature
    return gcn_feature
def get_gcn_feature_test_evo(cnn_feature,init_polys):
    if len(init_polys) == 0:
        return []
    img_poly = torch.tensor(init_polys).to(cnn_feature.device)
    batch_size,c,height, width = cnn_feature.shape
    img_poly = img_poly.float()
    img_poly[..., 0] = (img_poly[..., 0]  )/(width/2.0)  -  1
    img_poly[..., 1] = (img_poly[..., 1]   )/(height/2.0)  - 1 

    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(cnn_feature.device)
   
    for i in range(batch_size):
        img_poly = img_poly.unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], img_poly)[0].permute(1, 0, 2)
        gcn_feature = feature
    return gcn_feature

def get_gcn_feature_test(cnn_feature,center_boxes,init_polys,down_sample):
    center_boxes = torch.from_numpy(np.array(center_boxes)).to(cnn_feature.device)
    init_polys = torch.from_numpy(np.array(init_polys)).to(cnn_feature.device)
    batch_size,c,height, width = cnn_feature.shape
    ct = torch.ceil(center_boxes/down_sample).unsqueeze(1)
    init_polys = torch.ceil(init_polys/down_sample)
    points = torch.cat([ct,init_polys],dim=1)
    img_poly = points.float().clone()
    img_poly[..., 0] = img_poly[..., 0] / (width / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (height / 2.) - 1
    
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(cnn_feature.device)
    for i in range(batch_size):
        img_poly = img_poly.unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], img_poly)[0].permute(1, 0, 2)
        gcn_feature = feature

    return gcn_feature

def get_gcn_center_feature(cnn_feature,batch,down_sample):
    ind = batch['ct_img_idx'][batch['ct_01'].bool()]
    batch_size,c,height, width = cnn_feature.shape

    ct = torch.ceil(batch['center_boxes']/down_sample).unsqueeze(1)
 
    img_poly = ct.clone()
    img_poly[..., 0] = img_poly[..., 0] / (width / 2.) - 1
    img_poly[..., 1] = img_poly[..., 1] / (height / 2.) - 1
    
    gcn_feature = torch.zeros([img_poly.size(0), cnn_feature.size(1), img_poly.size(1)]).to(cnn_feature.device)
    for i in range(batch_size):
        poly = img_poly[ind == i].unsqueeze(0)
        feature = torch.nn.functional.grid_sample(cnn_feature[i:i+1], poly)[0].permute(1, 0, 2)
        gcn_feature[ind == i] = feature

    return gcn_feature
def Line_GeneralEquation(line=[1, 1, 1, 2]):
    A = line[3] - line[1]
    B = line[0] - line[2]
    C = line[2] * line[1] - line[0] * line[3]
    line = np.array([A, B, C])
    if B != 0:
        line = line / B
    return line
def SamplePointsOnLineSegment(point1, point2, distence):
    line_dist = np.sqrt((point2[0] - point1[0]) ** 2 + (point2[1] - point1[1]) ** 2)  
    num = round(line_dist / distence) 
    line = [point1[0], point1[1], point2[0], point2[1]]  
    line_ABC = Line_GeneralEquation(line)
    newP = []
    newP.append(point1)  
    if num > 0:
        dxy = line_dist / num 

        for i in range(1, num):
            if line_ABC[1] != 0:
                alpha = np.arctan(-line_ABC[0])
                dx = dxy * np.cos(alpha)
                dy = dxy * np.sin(alpha)
                if point2[0] - point1[0] > 0:
                    newP.append([point1[0] + i * dx, point1[1] + i * dy])
                else:
                    newP.append([point1[0] - i * dx, point1[1] - i * dy])
            else:
                if point2[1] - point1[1] > 0:
                    newP.append([point1[0], point1[1] + i * dxy])
                else:
                    newP.append([point1[0], point1[1] - i * dxy])
    newP.append([point2[0], point2[1]])  
    return np.array(newP)


def multiLine2Points(lineXY, distence):
   
    lineXY = np.array(lineXY)
    newPoints = []

    for i in range(len(lineXY) - 1):
        newP = SamplePointsOnLineSegment(lineXY[i, :], lineXY[i + 1, :], distence)
        newPoints.extend(newP)

    newP = SamplePointsOnLineSegment(lineXY[-1, :], lineXY[0, :], distence)
    newPoints.extend(newP)
    newPoints = np.array(newPoints)
    delInd = []
    for i in range(len(newPoints) - 1):
        if (newPoints[i, :] == newPoints[i + 1, :]).all():
            delInd.append(i)
    newPoints = np.delete(newPoints, delInd, axis=0)
   
    return newPoints

def calculate_polygon_area(points_list):
    n = len(points_list)
    if n < 3:
        return 0.0
    area = 0
    if n>4:
        step = int(n/4.0)
        points_list = points_list[0::step]
        n = len(points_list)
    for i in range(n):
        x = points_list[i][0]
        y = points_list[i][1]
        area += x * points_list[(i + 1) % n][1] - y * points_list[(i + 1) % n][0]
    return area * 0.5