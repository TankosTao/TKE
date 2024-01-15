import math
import numbers
import random

import cv2
import numpy as np
from skimage.util import random_noise


class RandomNoise:
    def __init__(self, random_rate):
        self.random_rate = random_rate

    def __call__(self, data: dict):
        """        :param data: {'img':,'text_polys':,'texts':,'ignore_tags':}
        :return:
        """
        if random.random() > self.random_rate:
            return data
        data['img'] = (random_noise(data['img'], mode='gaussian', clip=True) * 255).astype(im.dtype)
        return data


class RandomScale:
    def __init__(self, scales, random_rate):
     
        self.random_rate = random_rate
        self.scales = scales

    def __call__(self, data: dict) -> dict:
        
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        tmp_text_polys = text_polys.copy()
        rd_scale = float(np.random.choice(self.scales))
        im = cv2.resize(im, dsize=None, fx=rd_scale, fy=rd_scale)
        tmp_text_polys *= rd_scale

        data['img'] = im
        data['text_polys'] = tmp_text_polys
        return data



class RandomResize:
    def __init__(self, size, random_rate, keep_ratio=False):
        if isinstance(size, numbers.Number):
            if size < 0:
                raise ValueError("If input_size is a single number, it must be positive.")
            size = (size, size)
        elif isinstance(size, list) or isinstance(size, tuple) or isinstance(size, np.ndarray):
            if len(size) != 2:
                raise ValueError("If input_size is a sequence, it must be of len 2.")
            size = (size[0], size[1])
        else:
            raise Exception('input_size must in Number or list or tuple or np.ndarray')
        self.size = size
        self.keep_ratio = keep_ratio
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        if self.keep_ratio:
            h, w, c = im.shape
            max_h = max(h, self.size[0])
            max_w = max(w, self.size[1])
            im_padded = np.zeros((max_h, max_w, c), dtype=np.uint8)
            im_padded[:h, :w] = im.copy()
            im = im_padded
        text_polys = text_polys.astype(np.float32)
        h, w, _ = im.shape
        im = cv2.resize(im, self.size)
        w_scale = self.size[0] / float(w)
        h_scale = self.size[1] / float(h)
        text_polys[:, :, 0] *= w_scale
        text_polys[:, :, 1] *= h_scale

        data['img'] = im
        data['text_polys'] = text_polys
        return data


def resize_image(img, short_size):
    height, width, _ = img.shape
    new_width = short_size
    new_height = short_size
    new_height = int(round(new_height / 32) * 32)
    new_width = int(round(new_width / 32) * 32)
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img, (new_width / width, new_height / height)

class ResizeHW:
    def __init__(self, height=640, width=640,resize_text_polys=True):
        self.height = height
        self.width = width
        self.resize_text_polys = resize_text_polys
      

    def __call__(self, data: dict) -> dict:
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        fx_scale = self.width / w
        fy_scale = self.height/ h
        im = cv2.resize(im, dsize=None, fx=fx_scale, fy=fy_scale)
        if self.resize_text_polys:
            text_polys[:, 0] *= fx_scale
            text_polys[:, 1] *= fy_scale

        data['img'] = im
        data['text_polys'] = text_polys
        data['shape'] = [h,w]
        return data

class ResizeShortSize:
    def __init__(self, short_size,resize_height=True,resize_text_polys=True):
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys
        print(resize_height)
        self.resize_height = resize_height

    def __call__(self, data: dict) -> dict:

        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
        if self.resize_height:
            short_edge = h
        else:   
            short_edge = min(h, w)
        scale = self.short_size / short_edge
        
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        scale = (scale, scale)
        if self.resize_text_polys:
            text_polys[:, 0] *= scale[0]
            text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        data['shape'] = [h,w]
        return data
    
    
class ResizeHShortSize:
    def __init__(self, short_size, resize_text_polys=True):
        self.short_size = short_size
        self.resize_text_polys = resize_text_polys

    def __call__(self, data: dict) -> dict:
        im = data['img']
        text_polys = data['text_polys']

        h, w, _ = im.shape
    
        scale = self.short_size / h
        
        im = cv2.resize(im, dsize=None, fx=scale, fy=scale)
        scale = (scale, scale)
        if self.resize_text_polys:
            text_polys[:, 0] *= scale[0]
            text_polys[:, 1] *= scale[1]

        data['img'] = im
        data['text_polys'] = text_polys
        data['shape'] = [h,w]
        return data

class HorizontalFlip:
    def __init__(self, random_rate):
       
        self.random_rate = random_rate

    def __call__(self, data: dict) -> dict:
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 1)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 0] = w - flip_text_polys[:, :, 0]

        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data


class VerticallFlip:
    def __init__(self, random_rate):
        self.random_rate = random_rate
    def __call__(self, data: dict) -> dict:
        if random.random() > self.random_rate:
            return data
        im = data['img']
        text_polys = data['text_polys']

        flip_text_polys = text_polys.copy()
        flip_im = cv2.flip(im, 0)
        h, w, _ = flip_im.shape
        flip_text_polys[:, :, 1] = h - flip_text_polys[:, :, 1]
        data['img'] = flip_im
        data['text_polys'] = flip_text_polys
        return data
