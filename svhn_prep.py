#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Jul  5 12:19:07 2018

@author: mahyar
"""

import numpy as np
import scipy.io as sio
import digitStruct as ds
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
import cPickle as pk

'''
Read svhn uncut images and bbox max cut and resize them into im_data
im_data contains transformed images with 64*64 size
im_bboxes are transfored bboxes
im_names contain the name of images in the correct order
'''
svhn_test_path = '/media/evl/Public/Mahyar/Data/svhn/test'
svhn_train_path = '/media/evl/Public/Mahyar/Data/svhn/train'
counter = 0
#fig, ax = plt.subplots(1)
im_names = []
im_bboxes = []
im_data = []
im_size = 64

data_dir = svhn_train_path
### read svhn data struct one by one and fill im_data, im_boxes and im_names
for ds_obj in ds.yieldNextDigitStruct(data_dir+'/digitStruct.mat'):
    counter += 1
    ### read image
    im = Image.open(data_dir+'/'+ds_obj.name)
    im_names.append(ds_obj.name)
    w, h = im.size
    #print ds_obj.name
    #im_org = np.array(im, dtype=np.uint8)
    #ax.imshow(im_org)
    
    ### read bboxes and find convex hull
    min_left = min_top = 1000
    max_right = max_bottom = 0
    bbox_mat = []
    for bbox in ds_obj.bboxList:
        min_left = bbox.left if bbox.left < min_left else min_left
        min_top = bbox.top if bbox.top < min_top else min_top
        max_right = bbox.left+bbox.width if bbox.left+bbox.width > max_right else max_right
        max_bottom = bbox.top+bbox.height if bbox.top+bbox.height > max_bottom else max_bottom
        bbox_mat.append([bbox.left, bbox.top, bbox.left+bbox.width, bbox.top+bbox.height])
        #rect = patches.Rectangle((bbox.left, bbox.top), 
        #                         bbox.width, 
        #                         bbox.height, 
        #                         linewidth=1, edgecolor='r', facecolor='none')
        #ax.add_patch(rect)
    
    #plt.show()    
    bbox_mat = np.array(bbox_mat)
    
    ### extend bboxes convex hull to closest image edge
    cut_size = min(min_left, min_top, w-max_right, h-max_bottom)
    cut_size = cut_size if cut_size > 0 else 0
    left = max(min_left - cut_size, 0)
    top = max(min_top - cut_size, 0)
    right = min(max_right + cut_size, w)
    bottom = min(max_bottom + cut_size, h)
    
    ### crop and resize image using extended bboxes convex hull
    im_sq = im.crop((left, top, right, bottom))
    im_re_pil = im_sq.resize((im_size, im_size), Image.BILINEAR)
    im_data.append(np.array(im_re_pil, dtype=np.uint8))
    im_scale_w = 1.0 * im_size / im_sq.size[0]
    im_scale_h = 1.0 * im_size / im_sq.size[1]
    
    ### transform bboxes
    bbox_mat[:, 0] = np.maximum((bbox_mat[:, 0] - left) * im_scale_w, 0)
    bbox_mat[:, 1] = np.maximum((bbox_mat[:, 1] - top) * im_scale_h, 0)
    bbox_mat[:, 2] = np.minimum((bbox_mat[:, 2] - left) * im_scale_w, im_size-1)
    bbox_mat[:, 3] = np.minimum((bbox_mat[:, 3] - top) * im_scale_h, im_size-1)
    
    ### save transformed bboxes
    im_bboxes.append(np.array(bbox_mat, dtype=np.int32))
    '''
    ### draw
    fig, ax = plt.subplots(1)
    ax.imshow(im_data[-1])
    for i in range(im_bboxes[-1].shape[0]):
        bbox = im_bboxes[-1]
        rect = patches.Rectangle((bbox[i,0], bbox[i,1]), 
                                 bbox[i,2]-bbox[i,0], 
                                 bbox[i,3]-bbox[i,1], 
                                 linewidth=1, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
    plt.show()
    '''
    im.close()
    if counter % 100 == 0:
        print counter
    #if counter >= 50:
    #    break

### save to file
with open(data_dir+'/svhn_ndarray.cpk', 'wb+') as fs:
    pk.dump([im_bboxes, im_data, im_names], fs)


### read 32*32 images
'''
svhn_32_train = '/media/evl/Public/Mahyar/Data/svhn/train_32x32.mat'
svhn_32_test = '/media/evl/Public/Mahyar/Data/svhn/test_32x32.mat'
content_path = svhn_32_train
content_mat = sio.loadmat(content_path)
content_data = np.moveaxis(content_mat['X'], -1, 0)
content_labs = content_mat['y'].reshape((-1))
for i in range(content_data.shape[0]):
    fig, ax = plt.subplots(1)
    ax.imshow(content_data[i])
    plt.show()
    if i > 10:
        break
''' 
    
    
    