# -*- coding: utf-8 -*-
import os
import glob
import pathlib

data_path = r'test'

f_w = open(os.path.join(data_path, 'test.txt'), 'w', encoding='utf8')
for img_path in glob.glob(data_path + '/img/*.jpg', recursive=True):
    d = pathlib.Path(img_path)
    label_path = os.path.join(data_path, 'gt', ('gt_' + str(d.stem) + '.txt'))
    if os.path.exists(img_path) and os.path.exists(label_path):
        print(img_path, label_path)
    else:
        print('不存在', img_path, label_path)
    f_w.write('{}\t{}\n'.format(img_path, label_path))
f_w.close()