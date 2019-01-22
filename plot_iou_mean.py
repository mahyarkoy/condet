#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 16 16:41:15 2019

@author: mahyar
"""

import numpy as np
import matplotlib.pyplot as plt
import cPickle as pk
import matplotlib.cm as matcm
import os


### global colormap set
global_cmap = matcm.get_cmap('tab10')
global_color_locs = np.arange(10) / 10.
global_color_set = global_cmap(global_color_locs)

iou_paths = [
	'/media/evl/Public/Mahyar/condet_logs/8_logs_stn_rnbg_1shot/run_%d/iou_logs.cpk',
	'/media/evl/Public/Mahyar/condet_logs/9_logs_stn_rnbg_10shot/run_%d/iou_logs.cpk',
	'/media/evl/Public/Mahyar/condet_logs/10_logs_stn_rnbg_100shot/run_%d/iou_logs.cpk'
	]

def plot_iou_mean(ax, pathname, pname, pcolor):
	paths = list()
	### collect existing filnames for this pathname
	for i in range(10):
		try:
			p = pathname % i
		except:
			p = pathname
			paths.append(p)
			break
		if not os.path.exists(p):
			continue
		paths.append(p)
	### read iou_means
	print '>>> paths: ', paths
	iou_list = list()
	for p in paths:
		with open(p, 'rb') as fs:
			itrs_logs, iou_mat = pk.load(fs)
			iou_list.append(iou_mat[:, 0])
	ioum_mat = np.array(iou_list)
	ioum_mean = np.mean(ioum_mat, axis=0)
	ioum_std = np.std(ioum_mat, axis=0)
	### plot ioum means with std
	ax.plot(itrs_logs, ioum_mean, color=pcolor, label=pname)
	ax.plot(itrs_logs, ioum_mean+ioum_std, linestyle='--', linewidth=0.5, color=pcolor)
	ax.plot(itrs_logs, ioum_mean-ioum_std, linestyle='--', linewidth=0.5, color=pcolor)
	
if __name__ == '__main__':
	### prepare plot
	fig = plt.figure(0, figsize=(8,6))
	ax = fig.add_subplot(1,1,1)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_xlabel('Iterations')
	ax.set_ylabel('mean IOU')
	#ax.set_yscale('log')
	ax.set_title('Mean IOU')

	### plot
	pnames = ['1-shot', '10-shot', '100-shot']
	pcolors = [0, 1, 2]#, 3, 4, 5]
	for i, p in enumerate(iou_paths):
		plot_iou_mean(ax, p, pnames[i], global_color_set[pcolors[i]])
	
	ax.legend(loc=0)
	fig.savefig('/media/evl/Public/Mahyar/condet_logs/plots/iou_mean'+'_'.join(pnames)+'.pdf')