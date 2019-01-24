#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 11:10:34 2017

@author: mahyar
"""
### To convert to mp4 in command line
# ffmpeg -framerate 25 -i fields/field_%d.png -c:v libx264 -pix_fmt yuv420p baby_log_15.mp4
### To speed up mp4
# ffmpeg -i baby_log_57.mp4 -r 100 -filter:v "setpts=0.1*PTS" baby_log_57_100.mp4
# for i in {0..7}; do mv baby_log_a"$((i))" baby_log_"$((i+74))"; done

import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as mat_cm
import os
from progressbar import ETA, Bar, Percentage, ProgressBar
import argparse
print matplotlib.get_backend()
import cPickle as pk
import gzip
from skimage.transform import resize
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.utils.graph import graph_shortest_path
import sys
import scipy.io as sio
import skimage.io as skio
import glob
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', default='logs', help='log directory to store logs.')
arg_parser.add_argument('-e', '--eval', dest='eval_int', default=10000, help='eval intervals.')
arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
args = arg_parser.parse_args()
log_path = args.log_path
eval_int = int(args.eval_int)
run_seed = int(args.seed)

np.random.seed(run_seed)
tf.set_random_seed(run_seed)

import tf_condet

### global colormap set
global_cmap = mat_cm.get_cmap('tab10')
global_color_locs = np.arange(10) / 10.
global_color_set = global_cmap(global_color_locs)

'''
Generate noise background of im_size, and randomly paste co_data into it. pixel value is [0,1].
sample_size: if None the same number of samples as co_data are generated.
im_size w,h must be larger than co_data w,h
'''
def make_rand_bg(co_data, sample_size=None, im_size=(64, 64, 3)):
	co_size = co_data.shape[1:]
	sample_size = co_data.shape[0] if sample_size is None else sample_size
	
	### generate random bg
	#bgc = np.array([0., 0., 0.]).reshape((1,1,1,3))
	#im_bg = np.tile(bgc, (sample_size, im_size[0], im_size[1], 1))
	im_bg = np.random.uniform(size=(sample_size,)+im_size)

	### generate random bounding boxes with mnist digits
	im_bboxes = list()
	for i in range(sample_size):
		hs = np.random.choice([1, 2])
		ws = np.random.choice([1, 2])
		#hs = np.random.uniform(1., 2.)
		#ws = np.random.uniform(1., 2.)
		co_re = resize(co_data[i%sample_size, ...], (co_size[0]//hs, co_size[1]//ws), 
			anti_aliasing=True, preserve_range=True)
		co_re_size = co_re.shape
		top_rand = np.random.randint(im_size[0]-co_re_size[0])
		left_rand = np.random.randint(im_size[1]-co_re_size[1])
		#im_bg[i, ...] = add_clutter(im_bg[i, ...], co_data)
		im_bg[i, top_rand:top_rand+co_re_size[0], left_rand:left_rand+co_re_size[1], ...] = co_re
		im_bboxes.append(np.array(
			[left_rand, top_rand, left_rand+co_re_size[1]-1, top_rand+co_re_size[0]-1]).reshape(1,4))
	return im_bboxes, im_bg

def add_clutter(im, clutter_source):
	cl_size = (5, 5)
	cl_num = 5
	im_size = im.shape
	src_size = clutter_source.shape[1:]
	im_cl = np.array(im)
	for i in range(cl_num):
		top_rand = np.random.randint(im_size[0]-cl_size[0])
		left_rand = np.random.randint(im_size[1]-cl_size[1])
		top_rand_src = np.random.randint(src_size[0]-cl_size[0])
		left_rand_src = np.random.randint(src_size[1]-cl_size[1])
		id_src = np.random.randint(clutter_source.shape[0])
		sample_cl = clutter_source[id_src, top_rand_src:top_rand_src+cl_size[0], left_rand_src:left_rand_src+cl_size[1], ...]
		im_cl[top_rand:top_rand+cl_size[0], left_rand:left_rand+cl_size[1], ...] = sample_cl
	return im_cl


'''
Crop and resize single channel input (W,H) into im_size shape (PIL).
'''
def im_sq_resize(im_input, im_size, square=False):
	im = Image.fromarray(im_input, mode='F')
	im_sq = im
	if square:
		w, h = im.size
		im_cut = min(w, h)
		left = (w - im_cut) //2
		top = (h - im_cut) //2
		right = (w + im_cut) //2
		bottom = (h + im_cut) //2
		im_sq = im.crop((left, top, right, bottom))
	im_re_pil = im_sq.resize((im_size[1], im_size[0]), Image.BILINEAR)
	im_re = np.array(im_re_pil.getdata()).reshape(im_size)
	im.close()
	return im_re

'''
Get mnist [0,1] raw input data and return (bboxes, im_data, co_data) in [-1,1].
bboxes:	a list where each item is a matrix corresponding to one images with (l, t, r, b) as rows
label:	if None uses all of im_input to construct both co and im data.
		otherwise will select co_per_class images from each class of im_input as co_data
		and the rest as im_data.
warning: output data loses label info when lable is not None.
'''
def prep_mnist(im_input, co_size=(32, 32, 3), label=None, co_per_class=1):
	### read content images and resize to co_size
	im_input = im_input.reshape((im_input.shape[0], 28, 28, 1))
	im_input_re = np.zeros((im_input.shape[0], co_size[0], co_size[1], 1))
	for i in range(im_input.shape[0]):
		im_input_re[i, ...] = im_sq_resize(im_input[i, ..., 0], (co_size[0], co_size[1], 1))
			#resize(im_input[i, ...], (co_size[0], co_size[1]), preserve_range=True)
	co_data = np.tile(im_input_re, [1,1,1,3])

	### select a number of content images per class
	co_select = np.zeros(co_data.shape[0], dtype=bool)
	if label is not None:
		co_data, label = shuffle_data(co_data, label=label)
		class_num = np.max(label)+1
		class_counter = np.zeros(class_num)
		for i in range(co_data.shape[0]):
			if np.sum(class_counter) == class_num * co_per_class:
				break
			if class_counter[label[i]] == co_per_class:
				continue
			co_select[i] = True
			class_counter[label[i]] += 1
		im_select = np.invert(co_select)
	else:
		im_select = co_select = np.invert(co_select)

	### put co_data on random background
	bboxes, im_data = make_rand_bg(co_data[im_select, ...])
	return bboxes, im_data * 2. - 1., co_data[co_select] * 2. - 1.

def read_cub_file(fname):
	vals = list()
	with open(fname, 'r') as fs:
		for l in fs:
			vals.append(l.strip().split(' ')[1:])
	return vals

'''
Reading CUB dataset.
co_num, test_num, train_num: an ndarray of size 200, each element number of images to read from that class.
if integer is provided for any of the above, an ndarray with that value is contructed.
shuffles the read data.
'''
def read_cub(cub_path, co_num, test_num, train_num, im_size=128, co_size=64):
	### init
	co_num = np.array(co_num) if type(co_num) is np.ndarray else co_num*np.ones(200)
	train_num = np.array(train_num) if type(train_num) is np.ndarray else train_num*np.ones(200)
	test_num = np.array(test_num) if type(test_num) is np.ndarray else test_num*np.ones(200)

	### read image file names, bboxes, and classes
	im_fnames = np.array([v[0] for v in read_cub_file(cub_path+'/images.txt')])
	im_class = np.array([int(v[0]) for v in read_cub_file(cub_path+'/image_class_labels.txt')]) - 1
	im_bbox = np.array(
		[[int(float(v)) for v in bb] for bb in read_cub_file(cub_path+'/bounding_boxes.txt')])

	### shuffle
	order = np.arange(im_fnames.shape[0])
	np.random.shuffle(order)
	im_fnames = im_fnames[order]
	im_class = im_class[order]
	im_bbox = im_bbox[order, ...]

	### read images
	train_im = list()
	test_im = list()
	co_im = list()
	train_labs = list()
	test_labs = list()
	co_labs = list()
	train_bb = list()
	test_bb = list()
	total_num = np.sum(co_num+test_num+train_num)
	print '>>> Reading CUB from: '+cub_path
	widgets = ["CUB", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=total_num, widgets=widgets)
	pbar.start()
	counter = 0
	for i, c in enumerate(im_class):
		pbar.update(counter)
		counter += 1
		#print '>>> i: ', im_fnames[i]
		if co_num[c] > 0:
			im = read_image(cub_path+'/images/'+im_fnames[i], co_size, sqcrop=False, bbox=im_bbox[i])
			co_im.append(im)
			co_labs.append(c)
			co_num[c] -= 1
		elif test_num[c] > 0:
			im, w, h = read_image(cub_path+'/images/'+im_fnames[i], im_size, 
				sqcrop=False, verbose=True)
			test_im.append(im)
			test_labs.append(c)
			bbox = im_bbox[i]
			### transform bbox
			im_scale_w = 1.0 * im_size / w
			im_scale_h = 1.0 * im_size / h
			bbox[0] = int(bbox[0] * im_scale_w)
			bbox[1] = int(bbox[1] * im_scale_h)
			bbox[2] = int(bbox[2] * im_scale_w)
			bbox[3] = int(bbox[3] * im_scale_h)
			test_bb.append(np.array(
				[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]).reshape(1,4))
			test_num[c] -= 1	
		elif train_num[c] > 0:
			im, w, h = read_image(cub_path+'/images/'+im_fnames[i], im_size, 
				sqcrop=False, verbose=True)
			train_im.append(im)
			train_labs.append(c)
			bbox = im_bbox[i]
			### transform bbox
			im_scale_w = 1.0 * im_size / w
			im_scale_h = 1.0 * im_size / h
			bbox[0] = int(bbox[0] * im_scale_w)
			bbox[1] = int(bbox[1] * im_scale_h)
			bbox[2] = int(bbox[2] * im_scale_w)
			bbox[3] = int(bbox[3] * im_scale_h)
			train_bb.append(np.array(
				[bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]).reshape(1,4))
			train_num[c] -= 1
		else:
			counter -= 1
	return (np.array(co_im), co_labs), \
		(np.array(test_im), test_bb, test_labs), \
		(np.array(train_im), train_bb, train_labs)

def prep_svhn(co_data):
	co_data_re = (co_data + 1.0) / 2.0
	### put co_data on random background
	bboxes, im_data = make_rand_bg(co_data_re)
	return bboxes, im_data * 2. - 1.

'''
Reads svhn_prep data from file and return (bboxes, data, names) in [-1,1].
bboxes: a list where each item is a matrix corresponding to one images with (l, t, r, b) as rows
'''
def read_svhn_prep(path):
	### read data
	with open(path, 'rb') as fs:
		bboxes, im_data, im_names = pk.load(fs)
	return bboxes, np.array(im_data) * 2. / 255. - 1., im_names

'''
Reads svhn center and resize 32*32 images
'''
def read_svhn_32(path):
	content_mat = sio.loadmat(path)
	content_data = np.moveaxis(content_mat['X'], -1, 0)
	content_labs = content_mat['y'].reshape((-1))
	return content_data * 2. / 255. - 1., content_labs

'''
Reads mnist data from file and return (data, labels) for train, val, test respctively.
'''
def read_mnist(mnist_path):
	### read mnist data
	f = gzip.open(mnist_path, 'rb')
	train_set, val_set, test_set = pk.load(f)
	f.close()
	return train_set, val_set, test_set

'''
Resizes images to im_size and scale to (-1,1)
'''
def im_process(im_data, im_size=28):
	im_data = im_data.reshape((im_data.shape[0], 28, 28, 1))
	### resize
	#im_data_re = np.zeros((im_data.shape[0], im_size, im_size, 1))
	#for i in range(im_data.shape[0]):
	#	im_data_re[i, ...] = resize(im_data[i, ...], (im_size, im_size), preserve_range=True)
	im_data_re = np.array(im_data)

	### rescale
	im_data_re = im_data_re * 2.0 - 1.0
	return im_data_re

def read_image(im_path, im_size, sqcrop=True, bbox=None, verbose=False):
	im = Image.open(im_path)
	w, h = im.size
	if sqcrop:
		im_cut = min(w, h)
		left = (w - im_cut) //2
		top = (h - im_cut) //2
		right = (w + im_cut) //2
		bottom = (h + im_cut) //2
		im_sq = im.crop((left, top, right, bottom))
	elif bbox is not None:
		left = bbox[0]
		top = bbox[1]
		right = bbox[0] + bbox[2]
		bottom = bbox[1] + bbox[3]
		im_sq = im.crop((left, top, right, bottom))
	else:
		im_sq = im
	im_re_pil = im_sq.resize((im_size, im_size), Image.BILINEAR)
	im_re = np.array(im_re_pil.getdata())
	### next line is because pil removes the channels for black and white images!!!
	im_re = im_re if len(im_re.shape) > 1 else np.repeat(im_re[..., np.newaxis], 3, axis=1)
	im_re = im_re.reshape((im_size, im_size, 3))
	im.close()
	'''
	im = skio.imread(im_path)
	h, w, c = im.shape
	wc = w //2
	hc = h //2
	im_cut = min(wc, hc)
	im_sq = im[hc-im_cut:hc+im_cut, wc-im_cut:wc+im_cut, :]
	im_re = resize(im_sq, (im_size, im_size), preserve_range=True, anti_aliasing=True)
	'''
	im_o = im_re / 128.0 - 1.0 
	return im_o if not verbose else (im_o, w, h)

def read_lsun(lsun_path, data_size, im_size=64):
	im_data = np.zeros((data_size, im_size, im_size, 3))
	i = 0
	print '>>> Reading LSUN from: '+lsun_path
	widgets = ["LSUN", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=data_size, widgets=widgets)
	pbar.start()
	for fn in glob.glob(lsun_path+'/*.jpg'):
		pbar.update(i)
		im_data[i, ...] = read_image(fn, im_size)
		i += 1
		if i == data_size:
			break
	return im_data

def read_cifar(cifar_path):
	with open(cifar_path, 'rb') as fs:
		datadict = pk.load(fs)
	data = datadict['data'].reshape((-1, 3, 32, 32))
	labs = np.array(datadict['labels'])
	data_proc = data / 128.0 - 1.0
	return np.transpose(data_proc, axes=(0,2,3,1)), labs

def read_stl(stl_data_path, stl_lab_path, im_size=64):
	with open(stl_data_path, 'rb') as f:
		# read whole file in uint8 chunks
		everything = np.fromfile(f, dtype=np.uint8)
		im_data = np.reshape(everything, (-1, 3, 96, 96))
		im_data = np.transpose(im_data, (0, 3, 2, 1))

		### resize
		im_data_re = np.zeros((im_data.shape[0], im_size, im_size, 3))
		for i in range(im_data.shape[0]):
			im_data_re[i, ...] = resize(im_data[i, ...], (im_size, im_size), preserve_range=True)
		im_data_re = im_data_re / 128.0 - 1.0

	with open(stl_lab_path, 'rb') as f:
		labels = np.fromfile(f, dtype=np.uint8) - 1

	return im_data_re, labels

'''
Stacks images randomly on RGB channels, im_data shape must be (N, d, d, 1).
'''
def get_stack_mnist(im_data, labels=None, stack_size=3):
	order = np.arange(im_data.shape[0])
	
	np.random.shuffle(order)
	im_data_r = im_data[order, ...]
	labs_r = labels[order] if labels is not None else None

	if stack_size != 3:
		return im_data_r, labs_r

	np.random.shuffle(order)
	im_data_g = im_data[order, ...]
	labs_g = labels[order] if labels is not None else None

	np.random.shuffle(order)
	im_data_b = im_data[order, ...]
	labs_b = labels[order] if labels is not None else None

	### stack shuffled channels
	im_data_stacked = np.concatenate((im_data_r, im_data_g, im_data_b), axis=3)
	labs_stacked = labs_r + 10 * labs_g + 100 * labs_b if labels is not None else None
	
	return im_data_stacked, labs_stacked

def plot_time_series(name, vals, fignum, save_path, color='b', ytype='linear', itrs=None):
	fig, ax = plt.subplots(figsize=(8, 6))
	ax.clear()
	if itrs is None:
		ax.plot(vals, color=color)	
	else:
		ax.plot(itrs, vals, color=color)
	ax.grid(True, which='both', linestyle='dotted')
	ax.set_title(name)
	ax.set_xlabel('Iterations')
	ax.set_ylabel('Values')
	if ytype=='log':
		ax.set_yscale('log')
	fig.savefig(save_path, dpi=300)
	plt.close(fig)

def plot_time_mat(mat, mat_names, fignum, save_path, ytype=None, itrs=None):
	for n in range(mat.shape[1]):
		fig_name = mat_names[n]
		if not ytype:
			ytype = 'log' if 'param' in fig_name else 'linear'
		plot_time_series(fig_name, mat[:,n], fignum, save_path+'/'+fig_name+'.png', ytype=ytype, itrs=itrs)

'''
Samples sample_size images from each ganist generator, draws with color.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1]
'''
def gset_block_draw(ganist, sample_size, path, en_color=True, border=False):
	im_draw = np.zeros([ganist.g_num, sample_size]+ganist.data_dim)
	z_data = np.zeros(ganist.g_num*sample_size, dtype=np.int32)
	im_size = ganist.data_dim[0]
	for g in range(ganist.g_num):
		z_data[g*sample_size:(g+1)*sample_size] = g * np.ones(sample_size, dtype=np.int32)
		im_draw[g, ...] = sample_ganist(ganist, sample_size, z_data=z_data[g*sample_size:(g+1)*sample_size])
	#im_draw = (im_draw + 1.0) / 2.0
	if border:
		im_draw = im_color_borders(im_draw.reshape([-1]+ganist.data_dim), z_data)
		im_block_draw(im_draw, sample_size, path, z_data)
	elif en_color:
		en_block_draw(ganist, im_draw, path)
	else:
		block_draw(im_draw, path)

'''
Similar to gset_block_draw, except only draw high probability generators
'''
def gset_block_draw_top(ganist, sample_size, path, pr_th=0.05, en_color=False, g_color=True):
	g_pr = np.exp(ganist.pg_temp * ganist.g_rl_pvals)
	g_pr = g_pr / np.sum(g_pr)
	top_g_count = np.sum(g_pr > pr_th)
	im_draw = np.zeros([top_g_count, sample_size]+ganist.data_dim)
	z_data = np.zeros([top_g_count, sample_size], dtype=np.int32)
	im_size = ganist.data_dim[0]
	i = 0
	for g in range(ganist.g_num):
		if g_pr[g] <= pr_th:
			continue
		z_data[i, ...] = g * np.ones(sample_size, dtype=np.int32)
		im_draw[i, ...] = sample_ganist(ganist, sample_size, z_data=z_data[i, ...])
		i += 1
	#im_draw = (im_draw + 1.0) / 2.0
	if g_color is True:
		im_draw_flat = im_draw.reshape([-1]+ganist.data_dim)
		z_data_flat = z_data.reshape([-1])
		im_draw_color = im_color_borders(im_draw_flat, z_data_flat, max_label=ganist.g_num-1)
		im_draw = im_draw_color.reshape([top_g_count, sample_size]+ganist.data_dim[:-1]+[3])
	if en_color is True:
		en_block_draw(ganist, im_draw, path)
	else:
		block_draw(im_draw, path)

'''
Draws sample_size**2 randomly selected images from im_data.
If im_labels is provided: selects sample_size images for each im_label and puts in columns.
If ganist is provided: classifies selected images and adds color border.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_block_draw(im_data, sample_size, path, im_labels=None, ganist=None, border=False):
	imb, imh, imw, imc = im_data.shape
	if im_labels is not None:
		max_label = im_labels.max()
		im_draw = np.zeros([max_label+1, sample_size, imh, imw, imc])
		### select sample_size images from each label
		for g in range(max_label+1):
			im_draw[g, ...] = im_data[im_labels == g, ...][:sample_size, ...]
	else:
		draw_ids = np.random.choice(imb, size=sample_size**2, replace=False)
		im_draw = im_data[draw_ids, ...].reshape([sample_size, sample_size, imh, imw, imc])
	
	#im_draw = (im_draw + 1.0) / 2.0
	if ganist is not None:
		en_block_draw(ganist, im_draw, path)
	elif border:
		block_draw(im_draw, path, border=True)
	else:
		block_draw(im_draw, path)

'''
Classifies im_data with ganist e_net, draws with color borders.
im_data must have shape (cols, rows, imh, imw, imc) with values in [0,1]
'''
def en_block_draw(ganist, im_data, path, max_label=None):
	cols, rows, imh, imw, imc = im_data.shape
	max_label = ganist.g_num-1 if max_label is None else max_label
	im_draw_flat = im_data.reshape([-1]+ganist.data_dim)
	en_labels = np.argmax(eval_ganist_en(ganist, im_draw_flat), axis=1)
	im_draw_color = im_color_borders(im_draw_flat, en_labels, max_label=max_label)
	block_draw(im_draw_color.reshape([cols, rows, imh, imw, 3]), path)

'''
Adds a color border to im_data corresponding to its im_label.
im_data must have shape (imb, imh, imw, imc) with values in [-1,1].
'''
def im_color_borders(im_data, im_labels, max_label=None, color_map=None):
	fh = fw = 2
	imb, imh, imw, imc = im_data.shape
	max_label = im_labels.max() if max_label is None else max_label
	if imc == 1:
		im_data_t = np.tile(im_data, (1, 1, 1, 3))
	else:
		im_data_t = np.array(im_data)
	im_labels_norm = 1. * im_labels.reshape([-1]) / (max_label + 1)
	### pick rgb color for each label: (imb, 3) in [-1,1]
	if color_map is None:
		rgb_colors = global_color_set[im_labels, ...][:, :3] * 2. - 1.
	else:
		cmap = mat_cm.get_cmap(color_map)
		rgb_colors = cmap(im_labels_norm)[:, :3] * 2. - 1.
	rgb_colors_t = np.tile(rgb_colors.reshape((imb, 1, 1, 3)), (1, imh+2*fh, imw+2*fw, 1))

	### put im into rgb_colors_t
	for b in range(imb):
		rgb_colors_t[b, fh:imh+fh, fw:imw+fw, :] = im_data[b, ...]

	return rgb_colors_t

	### create mask
	#box_mask = np.ones((imh, imw))
	#box_mask[fh+1:imh-fh, fw+1:imw-fw] = 0.
	#box_mask_t = np.tile(box_mask.reshape((1, imh, imw, 1)), (imb, 1, 1, 3))
	#box_mask_inv = np.abs(box_mask_t - 1.)

	### apply mask
	#im_data_border = im_data_t * box_mask_inv + rgb_colors_t * box_mask_t
	#return im_data_border

'''
im_data should be a (columns, rows, imh, imw, imc).
im_data values should be in [0, 1].
If c is not 3 then draws first channel only.
'''
def block_draw(im_data, path, separate_channels=False, border=False):
	cols, rows, imh, imw, imc = im_data.shape
	### border
	if border:
		im_draw = im_color_borders(im_data.reshape((-1, imh, imw, imc)), 
			1.*np.ones(cols*rows, dtype=np.int32), max_label=0., color_map='RdBu')
		imb, imh, imw, imc = im_draw.shape
		im_draw = im_draw.reshape((cols, rows, imh, imw, imc))
	else:
		im_draw = im_data
	### block shape
	im_draw = im_draw.reshape([cols, imh*rows, imw, imc])
	im_draw = np.concatenate([im_draw[i, ...] for i in range(im_draw.shape[0])], axis=1)
	im_draw = (im_draw + 1.0) / 2.0
	### plots
	fig = plt.figure(0)
	fig.clf()
	if not separate_channels or im_draw.shape[-1] != 3:
		ax = fig.add_subplot(1, 1, 1)
		if im_draw.shape[-1] == 1:
			ims = ax.imshow(im_draw.reshape(im_draw.shape[:-1]))
		else:
			ims = ax.imshow(im_draw)
		ax.set_axis_off()
		#fig.colorbar(ims)
		fig.savefig(path, dpi=300)
	else:
		im_tmp = np.zeros(im_draw.shape)
		ax = fig.add_subplot(1, 3, 1)
		im_tmp[..., 0] = im_draw[..., 0]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		ax = fig.add_subplot(1, 3, 2)
		im_tmp[...] = 0.0
		im_tmp[..., 1] = im_draw[..., 1]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		ax = fig.add_subplot(1, 3, 3)
		im_tmp[...] = 0.0
		im_tmp[..., 2] = im_draw[..., 2]
		ax.set_axis_off()
		ax.imshow(im_tmp)

		fig.subplots_adjust(wspace=0, hspace=0)
		fig.savefig(path, dpi=300)

'''
Draws a row of images with bboxes top, corresponding attentions bottom.
ims: image tensor
bboxes: bbox matrix, row wise (l, t, r, b)
atts: attention tensor, same shape as ims
'''
def draw_im_att(ims, bboxes, path, trans=None, recs=None, atts=None):
	ims_bb = draw_bbox(ims, bboxes)
	### each column is one image info
	if trans is not None:
		atts_tile = np.tile(atts, [1, 1, 1, 3])
		im_mat = np.stack([ims_bb, recs, atts_tile * 2. - 1., trans, atts_tile*trans-(1.-atts_tile)], axis=1)
	else:
		im_mat = np.stack([ims_bb], axis=1)
	block_draw(im_mat, path, border=True)
	return

def draw_co(ims, trans, recs, path):
	im_mat = np.stack([ims, trans, recs], axis=1)
	block_draw(im_mat, path, border=True)
	return

def draw_im_stn(ims, bboxes, path, trans, recs, stn_bbox, stn_im):
	ims_bb = draw_bbox(ims, bboxes)
	ims_stn_bb = draw_bbox(ims, stn_bbox)
	stn_im_re = np.zeros(ims.shape)
	recs_re = np.zeros(ims.shape)
	trans_re = np.zeros(ims.shape)
	for i in range(ims.shape[0]):
		stn_im_re[i, ...] = resize(stn_im[i, ...], (ims.shape[1], ims.shape[2]), preserve_range=True)
	#for i in range(ims.shape[0]):
	#	recs_re[i, ...] = resize(recs[i, ...], (ims.shape[1], ims.shape[2]), preserve_range=True)
	#for i in range(ims.shape[0]):
	#	trans_re[i, ...] = resize(trans[i, ...], (ims.shape[1], ims.shape[2]), preserve_range=True)
	im_mat = np.stack([ims_bb, ims_stn_bb, stn_im_re], axis=1) #trans_re, recs_re], axis=1)
	block_draw(im_mat, path, border=True)
	return

'''
Draws bboxes on top of images.
ims: image tensor
bboxes: bbox matrix, row wise (l, t, r, b)
'''
def draw_bbox(ims, bboxes):
	lw = 2
	ims = np.array(ims)
	bcolor = 2.*np.array([1.0, 0.0, 0.0]) - 1.
	for i in range(ims.shape[0]):
		im = ims[i, ...]
		bbox_mat = bboxes[i]
		for b in range(bbox_mat.shape[0]):
			bbox = bbox_mat[b, ...]
			im[bbox[1]:bbox[3]+1, bbox[0]:bbox[0]+lw, ...] = bcolor
			im[bbox[1]:bbox[3]+1, bbox[2]-lw+1:bbox[2]+1, ...] = bcolor
			im[bbox[1]:bbox[1]+lw, bbox[0]:bbox[2]+1, ...] = bcolor
			im[bbox[3]-lw+1:bbox[3]+1, bbox[0]:bbox[2]+1, ...] = bcolor
	return ims

'''
Find bboxes from stn theta: shape (N, 6)
'''
def stn_theta_to_bbox(condet, theta):
	h, w = condet.data_dim[:2]
	#h = w = 64
	### bbox left and top
	bbox_l = (-theta[:,0] + theta[:, 2] + 1.0) * w / 2.0
	bbox_t = (-theta[:,4] + theta[:, 5] + 1.0) * h / 2.0
	bbox_l = np.where(bbox_l < 0, 0, bbox_l)
	bbox_t = np.where(bbox_t < 0, 0, bbox_t)
	bbox_l = np.where(bbox_l > w-1, w-1, bbox_l)
	bbox_t = np.where(bbox_t > h-1, h-1, bbox_t)

	### bbox right and bottom
	bbox_r = (theta[:,0] + theta[:, 2] + 1.0) * w / 2.0
	bbox_b = (theta[:,4] + theta[:, 5] + 1.0) * h / 2.0
	bbox_r = np.where(bbox_r < 0, 0, bbox_r)
	bbox_b = np.where(bbox_b < 0, 0, bbox_b)
	bbox_r = np.where(bbox_r > w-1, w-1, bbox_r)
	bbox_b = np.where(bbox_b > h-1, h-1, bbox_b)

	### reshape
	bbox_l = bbox_l.reshape((-1,1))
	bbox_t = bbox_t.reshape((-1,1))
	bbox_r = bbox_r.reshape((-1,1))
	bbox_b = bbox_b.reshape((-1,1))

	### concat and floor
	bbox = np.floor(np.concatenate((bbox_l, bbox_t, bbox_r, bbox_b), axis=1))
	bbox = bbox.astype(np.int32)
	return [bbox[b, ...].reshape((1,4)) for b in range(bbox.shape[0])]

def shuffle_data(im_data, im_bboxes=None, label=None):
	order = np.arange(im_data.shape[0])
	np.random.shuffle(order)
	im_data_sh = im_data[order, ...]
	if im_bboxes is None and label is None:
		return im_data_sh
	elif im_bboxes is None:
		return im_data_sh, label[order]
	elif label is None:
		im_bboxes_sh = [im_bboxes[i] for i in order]
		return im_data_sh, im_bboxes_sh
	else:
		im_bboxes_sh = [im_bboxes[i] for i in order]
		return im_data_sh, im_bboxes_sh, label[order]

'''
Train Condet:
co_data: content images
im_data: input images
im_bboxes: bounding boxes for each input image (list where each element row wise matrix of bbox coordinates)
'''
def train_condet(condet, im_data, co_data, im_bbox, test_im, test_bbox, labels=None):
	### dataset definition
	train_size = im_data.shape[0]

	### training configs
	max_itr_total = 5e5
	d_updates = 5
	g_updates = 1
	batch_size = 32
	eval_step = eval_int
	draw_step = eval_int

	### logs initi
	g_logs = list()
	d_r_logs = list()
	d_g_logs = list()
	eval_logs = list()
	eval_t_logs = list()
	stats_logs = list()
	itrs_logs = list()
	norms_logs = list()
	att_grad_mean_logs = list()
	g_loss_mean_logs = list()
	theta_logs = list()
	theta_stn_logs = list()
	theta_init_logs = list()

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	epoch = 0
	d_update_flag = True
	widgets = ["Condet", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	while itr_total < max_itr_total:
		### shuffle dataset
		train_im, train_bboxes = shuffle_data(im_data, im_bbox)
		train_co = shuffle_data(co_data)
		train_size = train_im.shape[0]
		co_size = train_co.shape[0]

		epoch += 1
		print ">>> Epoch %d started..." % epoch

		### train one epoch: input images size
		co_batch_start = 0
		for batch_start in range(0, train_size, batch_size):
			pbar.update(itr_total)
			### fetch batch data from input images
			batch_end = batch_start + batch_size
			batch_im = train_im[batch_start:batch_end, ...]
			batch_len = batch_im.shape[0]

			### choose content batch for each input batch
			co_batch_end = co_batch_start + batch_len
			batch_co = train_co[co_batch_start:co_batch_end, ...]
			if co_batch_end > co_size:
				co_len = batch_co.shape[0]
				co_batch_rep = [train_co for rep in range((batch_len-co_len)//co_size)]
				co_batch_rem = train_co[0:(batch_len-co_len)%co_size, ...]
				batch_co = np.concatenate([batch_co]+co_batch_rep+[co_batch_rem], axis=0)
				co_batch_start = (batch_len-co_len)%co_size
			else:
				co_batch_start += batch_len
		
			### evaluate energy distance between real and gen distributions
			if itr_total % eval_step == 0:
				draw_path = log_path_draw+'/sample_%d' % itr_total if itr_total % draw_step == 0 \
					else None
				iou_mean, iou_std, net_stats, g_theta, g_theta_stn, g_theta_init = eval_condet(condet, im_data, im_bbox, draw_path, co_data)
				iou_mean_t, iou_std_t, _, _, _, _ = eval_condet(condet, test_im, test_bbox)
				#e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
				eval_logs.append([iou_mean, iou_std])
				eval_t_logs.append([iou_mean_t, iou_std_t])
				stats_logs.append(net_stats)
				itrs_logs.append(itr_total)
				theta_logs.append([np.mean(g_theta, axis=0), np.std(g_theta, axis=0)])
				theta_stn_logs.append([np.mean(g_theta_stn, axis=0), np.std(g_theta_stn, axis=0)])
				theta_init_logs.append(g_theta_init)

				### norm logs
				grad_norms, d_r_loss_mean_sep, g_loss_mean_sep = condet.step(batch_im, batch_co, disc_only=True)
				norms_logs.append([np.max(grad_norms), np.mean(grad_norms), np.std(grad_norms)])
				att_grad_mean_logs.append(d_r_loss_mean_sep)
				g_loss_mean_logs.append(g_loss_mean_sep)
				
				### separate d_r loss mean plots
				#fig, ax = plt.subplots(figsize=(8, 6))
				#ax.clear()
				#ims = ax.matshow(d_r_loss_mean_sep.reshape((4,4)))
				#ax.set_title('d_r_loss mean')
				#fig.colorbar(ims)
				#fig.savefig(log_path_draw+'/sep_loss_%d_d_r.png' % itr_total, dpi=300)
				#plt.close(fig)
				
				### separate g loss mean plots
				#fig, ax = plt.subplots(figsize=(8, 6))
				#ax.clear()
				#ims = ax.matshow(g_loss_mean_sep.reshape((8,8)))
				#ax.set_title('g_loss mean')
				#fig.colorbar(ims)
				#fig.savefig(log_path_draw+'/sep_loss_%d_g.png' % itr_total, dpi=300)
				#plt.close(fig)

			### discriminator update
			if d_update_flag is True:
				batch_sum, batch_g_data = condet.step(batch_im, batch_co, gen_update=False)
				condet.write_sum(batch_sum, itr_total)
				d_itr += 1
				itr_total += 1
				d_update_flag = False if d_itr % d_updates == 0 else True

			### generator updates: g_updates times for each d_updates of discriminator
			elif g_updates > 0:
				pw = itr_total // 1000
				batch_sum, batch_g_data = condet.step(batch_im, batch_co, gen_update=True, run_count=pw)
				condet.write_sum(batch_sum, itr_total)
				g_itr += 1
				itr_total += 1
				d_update_flag = True if g_itr % g_updates == 0 else False

			if itr_total >= max_itr_total:
				break

		### save network every epoch
		condet.save(log_path_snap+'/model_%d_%d.h5' % (g_itr, itr_total))

		### plot condet evaluation plot every epoch **g_num**
		if len(eval_logs) < 2:
			continue
		eval_logs_mat = np.array(eval_logs)
		eval_t_logs_mat = np.array(eval_t_logs)
		stats_logs_mat = np.array(stats_logs)
		norms_logs_mat = np.array(norms_logs)
		att_grad_mean_mat = np.array(att_grad_mean_logs)
		g_loss_mean_mat = np.array(g_loss_mean_logs)
		theta_mat = np.array(theta_logs)
		theta_stn_mat = np.array(theta_stn_logs)
		theta_init_mat = np.array(theta_init_logs)

		#eval_logs_names = ['fid_dist', 'fid_dist']
		stats_logs_names = ['nan_vars_ratio', 'inf_vars_ratio', 'tiny_vars_ratio', 
							'big_vars_ratio']
		#plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path, itrs=itrs_logs)
		plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path, itrs=itrs_logs)
		
		### plot IOU
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		ax.plot(itrs_logs, eval_logs_mat[:,0], color='b', label='mean_iou')
		ax.plot(itrs_logs, eval_logs_mat[:,0]+eval_logs_mat[:,1], color='b', linestyle='--')
		ax.plot(itrs_logs, eval_logs_mat[:,0]-eval_logs_mat[:,1], color='b', linestyle='--')
		ax.set_ylim(bottom=0.0, top=1.0)
		ax.set_yticks(np.arange(0.0, 1.1, 0.1))
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Mean IOU')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/mean_iou_train.png', dpi=300)
		plt.close(fig)

		### plot IOU test
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		ax.plot(itrs_logs, eval_t_logs_mat[:,0], color='b', label='mean_iou')
		ax.plot(itrs_logs, eval_t_logs_mat[:,0]+eval_t_logs_mat[:,1], color='b', linestyle='--')
		ax.plot(itrs_logs, eval_t_logs_mat[:,0]-eval_t_logs_mat[:,1], color='b', linestyle='--')
		ax.set_ylim(bottom=0.0, top=1.0)
		ax.set_yticks(np.arange(0.0, 1.1, 0.1))
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Mean IOU Test')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/mean_iou_test.png', dpi=300)
		plt.close(fig)

		### plot norms
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		ax.plot(itrs_logs, norms_logs_mat[:,0], color='r', label='max_norm')
		ax.plot(itrs_logs, norms_logs_mat[:,1], color='b', label='mean_norm')
		ax.plot(itrs_logs, norms_logs_mat[:,1]+norms_logs_mat[:,2], color='b', linestyle='--')
		ax.plot(itrs_logs, norms_logs_mat[:,1]-norms_logs_mat[:,2], color='b', linestyle='--')
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Norm Grads')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/norm_grads.png', dpi=300)
		plt.close(fig)

		### plot theta
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(6):
			ax.plot(itrs_logs, theta_mat[:, 0, g], label='theta_%d' % g, c=global_color_set[g])
			ax.plot(itrs_logs, theta_mat[:, 0, g] + theta_mat[:, 1, g], c=global_color_set[g], linestyle='--', linewidth=0.5)
			ax.plot(itrs_logs, theta_mat[:, 0, g] - theta_mat[:, 1, g], c=global_color_set[g], linestyle='--', linewidth=0.5)
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Theta')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/theta_log.png', dpi=300)
		plt.close(fig)

		### plot theta_stn
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(6):
			ax.plot(itrs_logs, theta_stn_mat[:, 0, g], label='theta_stn_%d' % g, c=global_color_set[g])
			ax.plot(itrs_logs, theta_stn_mat[:, 0, g] + theta_stn_mat[:, 1, g], c=global_color_set[g], linestyle='--', linewidth=0.5)
			ax.plot(itrs_logs, theta_stn_mat[:, 0, g] - theta_stn_mat[:, 1, g], c=global_color_set[g], linestyle='--', linewidth=0.5)
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Theta STN')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/theta_stn_log.png', dpi=300)
		plt.close(fig)

		### plot theta_init
		fig, ax = plt.subplots(figsize=(8, 6))
		ax.clear()
		for g in range(6):
			ax.plot(itrs_logs, theta_init_mat[:, 0, g], label='theta_init_%d' % g, c=global_color_set[g])
		ax.grid(True, which='both', linestyle='dotted')
		ax.set_title('Theta INIT')
		ax.set_xlabel('Iterations')
		ax.set_ylabel('Values')
		ax.legend(loc=0)
		fig.savefig(log_path+'/theta_init_log.png', dpi=300)
		plt.close(fig)

	### save eval_logs
	with open(log_path+'/iou_logs.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, eval_logs_mat], fs)
	### save eval_t_logs
	with open(log_path+'/iou_test_logs.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, eval_t_logs_mat], fs)

'''
Sample sample_size data points from ganist.
'''
def sample_ganist(ganist, sample_size, sampler=None, batch_size=64, 
	z_data=None, zi_data=None, z_im=None):
	sampler = sampler if sampler is not None else ganist.step
	g_samples = np.zeros([sample_size] + ganist.data_dim)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_len = g_samples[batch_start:batch_end, ...].shape[0]
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_zi = zi_data[batch_start:batch_end, ...] if zi_data is not None else None
		batch_im = z_im[batch_start:batch_end, ...] if z_im is not None else None
		g_samples[batch_start:batch_end, ...] = \
			sampler(batch_im, batch_len, gen_only=True, z_data=batch_z, zi_data=batch_zi)
	return g_samples

'''
Random box baseline
'''
def rand_baseline_att(im_data, wsize=32):
	im_att = np.zeros(im_data.shape[:3]+(1,))
	for b in range(im_data.shape[0]):
		h, w, _ = im_data[b, ...].shape
		ht = np.random.randint(h-wsize)
		wl = np.random.randint(w-wsize)
		im_att[b, ht:ht+wsize, wl:wl+wsize, ...] = 1.0
	return im_att

'''
Generate attention.
if att_co is false, im_data passes through the gen transformation.
'''
def condet_att(condet, im_data, batch_size=64, att_co=False):
	#im_att = np.zeros(im_data.shape[:3]+(1,))
	im_att = np.zeros([im_data.shape[0]]+condet.co_dim)
	im_trans = np.zeros([im_data.shape[0]]+condet.co_dim)
	im_rec = np.zeros([im_data.shape[0]]+condet.co_dim)
	im_theta = np.zeros([im_data.shape[0], 6])
	im_theta_stn = np.zeros([im_data.shape[0], 6])
	im_theta_init = np.zeros([1, 6])
	for batch_start in range(0, im_data.shape[0], batch_size):
		batch_end = batch_start + batch_size
		batch_im = im_data[batch_start:batch_end, ...]
		if not att_co:
			im_trans[batch_start:batch_end, ...], im_rec[batch_start:batch_end, ...], \
			im_att[batch_start:batch_end, ...], im_theta[batch_start:batch_end, ...], \
			im_theta_stn[batch_start:batch_end, ...], im_theta_init[...]= \
				condet.step(batch_im, att_only_im=True)
		else:
			im_trans[batch_start:batch_end, ...], im_rec[batch_start:batch_end, ...] = condet.step(batch_im, att_only_co=True)
	return im_trans, im_rec, im_att, im_theta, im_theta_stn, im_theta_init

'''
Generate transformation.
'''
def condet_trans(condet, im_data, batch_size=64):
	im_trans = np.zeros(im_data.shape)
	sample_size = im_data.shape[0]
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_im = im_data[batch_start:batch_end, ...]
		im_trans[batch_start:batch_end, ...] = condet.step(batch_im, gen_only=True)
	return im_trans

'''
Run discriminator of ganist on the given im_data, return logits and gradient norms. **g_num**
'''
def run_ganist_disc(ganist, im_data, sampler=None, batch_size=64, z_data=None):
	sampler = sampler if sampler is not None else ganist.step
	sample_size = im_data.shape[0]
	logits = np.zeros(sample_size)
	grad_norms = np.zeros(sample_size)
	for batch_start in range(0, sample_size, batch_size):
		batch_end = batch_start + batch_size
		batch_z = z_data[batch_start:batch_end, ...] if z_data is not None else None
		batch_im = im_data[batch_start:batch_end, ...]
		batch_logits, batch_grad_norms = sampler(batch_im, None, dis_only=True, z_data=batch_z)
		logits[batch_start:batch_end] = batch_logits
		grad_norms[batch_start:batch_end] = batch_grad_norms
	return logits, grad_norms

'''
Evaluate intersection over union (mean and std over inputs)
'''
def eval_iou(atts, bboxes):
	bbox_im = bbox_to_att(bboxes, atts.shape)
	bbox_sum = np.sum(bbox_im, axis=(1,2,3))
	atts_sum = np.sum(atts, axis=(1,2,3))
	inter_sum = np.sum(atts*bbox_im, axis=(1,2,3))
	uni_sum = bbox_sum + atts_sum - inter_sum
	iou = inter_sum / uni_sum
	return np.mean(iou), np.std(iou)

'''
Convert bboxes to atts
'''
def bbox_to_att(bboxes, im_size):
	bbox_im = np.zeros(im_size)
	for i, bbox_mat in enumerate(bboxes):
		for b in range(bbox_mat.shape[0]):
			bbox = bbox_mat[b]
			bbox_im[i, bbox[1]:bbox[3], bbox[0]:bbox[2], ...] = 1.0
	return bbox_im

'''
Returns intersection over union mean and std, net_stats, and draw_im_att
'''
def eval_condet(condet, im_data, bboxes, draw_path=None, co_data=None, sample_size=1000):
	### sample and batch size
	batch_size = 64
	draw_size = 20
	co_data = None
	
	### collect real and gen samples **mt**
	r_samples = im_data[0:sample_size, ...]
	r_bboxes = bboxes[0:sample_size]
	g_samples, g_rec, g_att, g_theta, g_theta_stn, g_theta_init = condet_att(condet, r_samples)
	g_stn_bbox = stn_theta_to_bbox(condet, g_theta)

	### draw block image of gen samples
	if draw_path is not None:
		draw_im_stn(r_samples[0:draw_size, ...], r_bboxes[0:draw_size], draw_path+'_im.png', 
			g_samples[0:draw_size, ...], g_rec[0:draw_size, ...], 
			g_stn_bbox[0:draw_size], g_att[0:draw_size, ...])
		if co_data is not None:
			co_samples = co_data[0:draw_size, ...]
			co_g_samples, co_rec, _, _, _, _ = condet_att(condet, co_samples, att_co=True)
			draw_co(co_samples, co_g_samples, co_rec, draw_path+'_co.png')
		#draw_im_att(r_samples[0:draw_size, ...], r_bboxes[0:draw_size], draw_path, 
		#	g_samples[0:draw_size, ...], g_rec[0:draw_size, ...], g_att[0:draw_size, ...])

	### get network stats
	net_stats = condet.step(None, stats_only=True)

	### iou
	g_att_stn = bbox_to_att(g_stn_bbox, r_samples.shape)
	iou_mean, iou_std = eval_iou(g_att_stn, r_bboxes)

	return iou_mean, iou_std, net_stats, g_theta, g_theta_stn, g_theta_init


if __name__ == '__main__':
	### log path setups
	mnist_stack_size = 1
	log_path_snap = log_path+'/snapshots'
	log_path_draw = log_path+'/draws'
	log_path_sum = log_path+'/sums'

	os.system('mkdir -p '+log_path_snap)
	os.system('mkdir -p '+log_path_draw)
	os.system('mkdir -p '+log_path_sum)
	'''
	DATASET LOADING AND DRAWING
	'''
	### mnist dataset
	'''
	train_data, val_data, test_data = read_mnist(data_path)
	train_labs = train_data[1]
	train_imgs = im_process(train_data[0])
	val_labs = val_data[1]
	val_imgs = im_process(val_data[0])
	test_labs = test_data[1]
	test_imgs = im_process(test_data[0])
	all_labs = np.concatenate([train_labs, val_labs, test_labs], axis=0)
	all_imgs = np.concatenate([train_imgs, val_imgs, test_imgs], axis=0)
	'''
	### svhn_uncut (as input images)
	'''
	svhn_test_path = '/media/evl/Public/Mahyar/Data/svhn/test/svhn_ndarray.cpk'
	svhn_train_path = '/media/evl/Public/Mahyar/Data/svhn/train/svhn_ndarray.cpk'
	train_bbox, train_im, train_names = read_svhn_prep(svhn_train_path)
	test_bbox, test_im, test_names = read_svhn_prep(svhn_test_path)
	print '>>> INPUT TRAIN SIZE:', train_im.shape
	print '>>> INPUT TEST SIZE:', test_im.shape
	'''
	### svhn_cut (as content images)
	'''
	svhn_32_train = '/media/evl/Public/Mahyar/Data/svhn/train_32x32.mat'
	svhn_32_test = '/media/evl/Public/Mahyar/Data/svhn/test_32x32.mat'
	svhn_train_co, svhn_train_co_labs = read_svhn_32(svhn_32_train)
	svhn_test_co, svhn_test_co_labs = read_svhn_32(svhn_32_test)
	svhn_train_bbox, svhn_train_im = prep_svhn(svhn_train_co)
	svhn_test_bbox, svhn_test_im = prep_svhn(svhn_test_co)
	print '>>> SVHN TRAIN SIZE:', svhn_train_co.shape
	print '>>> SVHN TEST SIZE:', svhn_test_co.shape
	'''

	### mnist with noise background
	#mnist_path = '/media/evl/Public/Mahyar/Data/mnist.pkl.gz'
	#mnist_train_data, mnist_val_data, mnist_test_data = read_mnist(mnist_path)
	#mnist_train_bbox, mnist_train_im, mnist_train_co = prep_mnist(mnist_train_data[0])
	#mnist_val_bbox, mnist_val_im, mnist_val_co = prep_mnist(mnist_val_data[0], label=mnist_val_data[1], co_per_class=1)
	#mnist_test_bbox, mnist_test_im, mnist_test_co = prep_mnist(mnist_test_data[0])
	#print '>>> MNIST TRAIN SIZE:', mnist_train_im.shape
	#print '>>> MNIST TEST SIZE:', mnist_test_im.shape
	
	### dataset choice
	#train_im = mnist_train_im
	#train_bbox = mnist_train_bbox
	#train_co = mnist_val_co
	
	#test_im = mnist_test_im
	#test_bbox = mnist_test_bbox

	### cub data
	co_num = 10
	test_num = 5
	train_num = 50
	cub_path = '/media/evl/Public/Mahyar/Data/cub/CUB_200_2011'
	cub_co_data, cub_test_data, cub_train_data = read_cub(cub_path, co_num, test_num, train_num)
	print '>>> CUB CO SIZE:', cub_co_data[0].shape
	print '>>> CUB TEST SIZE:', cub_test_data[0].shape
	print '>>> CUB TRAIN SIZE:', cub_train_data[0].shape

	### dataset choice
	train_im = cub_train_data[0]
	train_bbox = cub_train_data[1]
	train_co = cub_co_data[0]

	test_im = cub_test_data[0]
	test_bbox = cub_test_data[1]

	'''
	TENSORFLOW SETUP
	'''
	gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.95)
	config = tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)
	sess = tf.Session(config=config)
	
	### create a condet instance
	condet = tf_condet.Condet(sess, log_path_sum)
	
	### init variables
	sess.run(tf.global_variables_initializer())
	
	### save network initially
	condet.save(log_path_snap+'/model_0_0.h5')
	with open(log_path+'/vars_count_log.txt', 'w+') as fs:
		print >>fs, '>>> g_vars: %d --- d_vars: %d --- a_vars: %d --- i_vars: %d' \
			% (condet.g_vars_count, condet.d_vars_count, condet.a_vars_count, condet.i_vars_count)
	

	#draw_im_att(test_im[:20], test_bbox[:20], path=log_path+'/im_bb.png')

	### random att iou
	#sample_size = 1000
	#rand_att = rand_baseline_att(test_im[0:sample_size], wsize=32)
	#draw_im_att(test_im[0:20, ...], test_bbox[0:20], log_path+'/rand_sample.png',
	#	trans=test_im[0:20, ...], recs=test_im[0:20, ...], atts=rand_att[0:20, ...])
	#iou_mean, iou_std = eval_iou(rand_att, test_bbox[0:sample_size])
	#print ">>> Rand IOU mean: ", iou_mean
	#print ">>> Rand IOU std: ", iou_std

	### draw samples from content
	print ">>> TRAIN CO shape: ", train_co.shape
	print ">>> TRAIN IM shape: ", train_im.shape
	block_draw(train_co[:20].reshape((20, 1)+train_co.shape[1:]), log_path+'/train_co_10.png', border=True)
	im_block_draw(train_co, 5, log_path+'/train_co_samples.png', border=True)

	'''
	GAN SETUP SECTION
	'''
	### train condet
	train_condet(condet, train_im, train_co, train_bbox, test_im, test_bbox)

	### load condet
	# condet.load(condet_path % run_seed)

	'''
	GAN DATA EVAL
	'''
	draw_path = log_path+'/sample_final.png'
	iou_mean, iou_std, net_stats, _, _, _ = eval_condet(condet, test_im, test_bbox, draw_path)
	print ">>> IOU mean: ", iou_mean
	print ">>> IOU std: ", iou_std
	with open(log_path+'/iou_test_log.txt', 'w+') as fs:
		print >>fs, '>>> iou_mean: %f --- iou_std: %f' \
			% (iou_mean, iou_std)
	sess.close()

