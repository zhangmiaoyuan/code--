# -*- coding: utf-8 -*-
import csv

import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import scipy.spatial
import json
from matplotlib import cm as CM
import torch
import os
from xml.dom.minidom import parse
import numpy as np


root = './'
train = os.path.join(root,'train_img')
#
def readcsv(filename):
    my_list = []
    with open(os.path.join(root, filename), mode='r') as f:
        reader = csv.reader(f)
        for row in reader:
            pos_x = int(row[1])
            pos_y = int(row[2])
            # print('pos_x:',pos_x)
            my_list.append([pos_x, pos_y])
        print('len:',len(my_list))
        print(my_list)
    return np.array(my_list)
list = readcsv('0.csv')


# import csv
#
# my_list = []
# with open('0.csv', 'r') as csv_file:
#     reader = csv.reader(csv_file)
#     print(reader)
#     for row in reader:
#         print(row)
#         pos_x = float(row[1])
#         pos_y = float(row[2])
#         my_list.append([pos_x, pos_y])
#
# print(my_list)
image_path = []
image_path += glob.glob(os.path.join(root,  '*.jpg'))
print(image_path)
print(image_path[0])

for i , image_path in enumerate(image_path):
    gt = readcsv(image_path.replace('.jpg','.csv').replace('Infrared','GT_Infrared'))
    img = plt.imread(image_path)
    k = np.zeros((img.shape[0],img.shape[1]))
    print(gt.shape)
    print(k.shape)
    print(int(gt[0][1]))
    for i in range(0,len(gt)):
        if int(gt[i][1])<img.shape[0]and int(gt[i][0]) < img.shape[1]:
            k[int(gt[i][1]), int(gt[i][0])] = 1
    k = gaussian_filter(k, 15)
    with h5py.File(image_path.replace('.jpg', '.h5').replace('img', 'GT_img'), 'w') as hf:
        hf['density'] = k
    '''----------------------------生成.npy文件------------------------------'''
    np.save(image_path.replace('.jpg', '.npy').replace('img', 'GT_img'), k)
plt.imshow(Image.open(image_path))
plt.show()
# 保存并可视化密度图
gt_file = h5py.File(image_path.replace('.jpg', '.h5').replace('img', 'GT_img'), 'r')
groundtruth = np.asarray(gt_file['density'])
plt.imsave("./" + image_path[0] +"00.jpg", groundtruth, cmap=CM.jet)
plt.imshow(groundtruth, cmap=CM.jet)
plt.show()
print(np.sum(groundtruth))  # 查看一共有多少人，一般情况下会和真实情况有些许出入

