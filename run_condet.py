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
import scipy
import skimage.io as skio
import glob
from PIL import Image

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument('-l', '--log-path', dest='log_path', required=True, help='log directory to store logs.')
arg_parser.add_argument('-e', '--eval', dest='eval_int', required=True, help='eval intervals.')
arg_parser.add_argument('-s', '--seed', dest='seed', default=0, help='random seed.')
args = arg_parser.parse_args()
log_path = args.log_path
eval_int = int(args.eval_int)
run_seed = int(args.seed)

np.random.seed(run_seed)
tf.set_random_seed(run_seed)

import tf_condet

### log path setups
mnist_stack_size = 1
log_path_snap = log_path+'/snapshots'
log_path_draw = log_path+'/draws'
log_path_sum = log_path+'/sums'

os.system('mkdir -p '+log_path_snap)
os.system('mkdir -p '+log_path_draw)
os.system('mkdir -p '+log_path_sum)

'''
Reads svhn_prep data from file and return (bboxes, data).
'''
def read_svhn_prep(path):
	### read data
	with open(path, 'rb') as fs:
		bboxes, im_data, im_names = pk.load(fs)
	return bboxes, im_data, im_names

'''
Reads svhn center and resize 32*32 images
'''
def read_svhn_32(path):
	content_mat = sio.loadmat(path)
	content_data = np.moveaxis(content_mat['X'], -1, 0)
	content_labs = content_mat['y'].reshape((-1))
	return content_data, content_labs

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

def read_image(im_path, im_size):
	im = Image.open(im_path)
	w, h = im.size
	im_cut = min(w, h)
	left = (w - im_cut) //2
	top = (h - im_cut) //2
	right = (w + im_cut) //2
	bottom = (h + im_cut) //2
	im_sq = im.crop((left, top, right, bottom))
	im_re_pil = im_sq.resize((im_size, im_size), Image.BILINEAR)
	im_re = np.array(im_re_pil.getdata()).reshape((im_size, im_size, 3))
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
	return im_re / 128.0 - 1.0

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
	rgb_colors_t = np.tile(rgb_colors.reshape((imb, 1, 1, 3)), (1, imh, imw, 1))

	### create mask
	box_mask = np.ones((imh, imw))
	box_mask[fh+1:imh-fh, fw+1:imw-fw] = 0.
	box_mask_t = np.tile(box_mask.reshape((1, imh, imw, 1)), (imb, 1, 1, 3))
	box_mask_inv = np.abs(box_mask_t - 1.)

	### apply mask
	im_data_border = im_data_t * box_mask_inv + rgb_colors_t * box_mask_t
	return im_data_border

'''
im_data should be a (columns, rows, imh, imw, imc).
im_data values should be in [0, 1].
If c is not 3 then draws first channel only.
'''
def block_draw(im_data, path, separate_channels=False, border=False):
	cols, rows, imh, imw, imc = im_data.shape
	### border
	if border:
		im_draw = im_color_borders(im_data.reshape((-1, imh, imw, imc)), np.zeros(cols*rows, dtype=np.int32), color_map='hot')
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

def shuffle_data(im_data, im_bboxes=None):
	order = np.arange(im_data.shape[0])
	np.random.shuffle(order)
	im_data_sh = im_data[order, ...]
	if im_bboxes is None:
		return im_data_sh
	else:
		im_bboxes_sh = im_bboxes[order, ...]
		return im_data_sh, im_bboxes_sh

'''
Train Ganist
'''
def train_condet(condet, co_data, im_data, im_bboxes, labels=None):
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
	stats_logs = list()
	norms_logs = list()
	itrs_logs = list()
	rl_vals_logs = list()
	rl_pvals_logs = list()
	en_acc_logs = list()

	### training inits
	d_itr = 0
	g_itr = 0
	itr_total = 0
	epoch = 0
	d_update_flag = True
	widgets = ["Ganist", Percentage(), Bar(), ETA()]
	pbar = ProgressBar(maxval=max_itr_total, widgets=widgets)
	pbar.start()

	while itr_total < max_itr_total:
		### shuffle dataset
		train_im, train_bboxes = shuffle_data(im_data, im_bbox)
		train_co = shuffle_data(co_data)
		train_size = min(train_im.shape[0], train_co.shape[0])

		epoch += 1
		print ">>> Epoch %d started..." % epoch

		### train one epoch
		for batch_start in range(0, train_size, batch_size):
			pbar.update(itr_total)
			batch_end = batch_start + batch_size
			### fetch batch data
			batch_co = train_co[batch_start:batch_end, ...]
			batch_im = train_im[batch_start:batch_end, ...]
			fetch_batch = False
		
			### evaluate energy distance between real and gen distributions
			if itr_total % eval_step == 0:
				draw_path = log_path_draw+'/gen_sample_%d' % itr_total if itr_total % draw_step == 0 \
					else None
				e_dist, fid_dist, net_stats = eval_ganist(ganist, train_dataset, draw_path)
				#e_dist = 0 if e_dist < 0 else np.sqrt(e_dist)
				eval_logs.append([e_dist, fid_dist])
				stats_logs.append(net_stats)
				### log norms every epoch
				d_sample_size = 100
				_, grad_norms = run_ganist_disc(ganist, 
					train_dataset[0:d_sample_size, ...], batch_size=256)
				norms_logs.append([np.max(grad_norms), np.mean(grad_norms), np.std(grad_norms)])
				itrs_logs.append(itr_total)

				### log rl vals and pvals **g_num**
				rl_vals_logs.append(list(ganist.g_rl_vals))
				rl_pvals_logs.append(list(ganist.g_rl_pvals))
				#z_pr = np.exp(ganist.pg_temp * ganist.g_rl_pvals)
				#z_pr = z_pr / np.sum(z_pr)
				#rl_pvals_logs.append(list(z_pr))

				### en_accuracy plots **g_num**
				acc_array = np.zeros(ganist.g_num)
				sample_size = 1000
				for g in range(ganist.g_num):
					z = g * np.ones(sample_size)
					z = z.astype(np.int32)
					g_samples = sample_ganist(ganist, sample_size, z_data=z)
					acc_array[g] = eval_en_acc(ganist, g_samples, z)
				en_acc_logs.append(list(acc_array))

				### draw real samples en classified **g_num**
				d_sample_size = 1000
				#im_true_color = im_color_borders(train_dataset[:d_sample_size], 
				#	train_labs[:d_sample_size], max_label=9)
				#im_block_draw(im_true_color, 10, draw_path+'_t.png', 
				#	im_labels=train_labs[:d_sample_size])
				im_block_draw(train_dataset[:d_sample_size], 10, draw_path+'_t.png', 
					im_labels=train_labs[:d_sample_size], ganist=ganist)

			### discriminator update
			if d_update_flag is True:
				batch_sum, batch_g_data = condet.step(batch_co, batch_im, gen_update=False)
				condet.write_sum(batch_sum, itr_total)
				d_itr += 1
				itr_total += 1
				d_update_flag = False if d_itr % d_updates == 0 else True

			### generator updates: g_updates times for each d_updates of discriminator
			elif g_updates > 0:
				batch_sum, batch_g_data = condet.step(batch_co, gen_update=True)
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
		stats_logs_mat = np.array(stats_logs)
		norms_logs_mat = np.array(norms_logs)

		#eval_logs_names = ['fid_dist', 'fid_dist']
		stats_logs_names = ['nan_vars_ratio', 'inf_vars_ratio', 'tiny_vars_ratio', 
							'big_vars_ratio']
		#plot_time_mat(eval_logs_mat, eval_logs_names, 1, log_path, itrs=itrs_logs)
		plot_time_mat(stats_logs_mat, stats_logs_names, 1, log_path, itrs=itrs_logs)
		
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

	### save norm_logs
	with open(log_path+'/norm_grads.cpk', 'wb+') as fs:
		pk.dump(norms_logs_mat, fs)

	### save pval_logs
	with open(log_path+'/rl_pvals.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, rl_pvals_logs_mat], fs)

	### save eval_logs
	with open(log_path+'/eval_logs.cpk', 'wb+') as fs:
		pk.dump([itrs_logs, eval_logs_mat], fs)

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
Returns the energy distance of a trained GANist, and draws block images of GAN samples
'''
def eval_condet(ganist, im_data, draw_path=None, sampler=None):
	### sample and batch size
	sample_size = 10000
	batch_size = 64
	draw_size = 10
	sampler = sampler if sampler is not None else ganist.step
	
	### collect real and gen samples **mt**
	r_samples = im_data[0:sample_size, ...]
	g_samples = sample_ganist(ganist, sample_size, sampler=sampler,
		z_im=im_data[-sample_size:, ...])
	
	### calculate energy distance
	#rr_score = np.mean(np.sqrt(np.sum(np.square( \
	#	r_samples[0:sample_size//2, ...] - r_samples[sample_size//2:, ...]), axis=1)))
	#gg_score = np.mean(np.sqrt(np.sum(np.square( \
	#	g_samples[0:sample_size//2, ...] - g_samples[sample_size//2:, ...]), axis=1)))
	#rg_score = np.mean(np.sqrt(np.sum(np.square( \
	#	r_samples[0:sample_size//2, ...] - g_samples[0:sample_size//2, ...]), axis=1)))

	### draw block image of gen samples
	if draw_path is not None:
		g_samples = g_samples.reshape((-1,) + im_data.shape[1:])
		### manifold interpolation drawing mode **mt** **g_num**
		'''
		gr_samples = im_data[-sample_size:, ...]
		gr_flip = np.array(gr_samples)
		for batch_start in range(0, sample_size, batch_size):
			batch_end = batch_start + batch_size
			gr_flip[batch_start:batch_end, ...] = np.flip(gr_flip[batch_start:batch_end, ...], axis=0)
		draw_samples = np.concatenate([g_samples, gr_samples, gr_flip], axis=3)
		im_block_draw(draw_samples, draw_size, draw_path)
		'''
		### **g_num**
		im_block_draw(g_samples, draw_size, draw_path+'.png', ganist=ganist)
		gset_block_draw(ganist, 10, draw_path+'_gset.png', en_color=True)

	### get network stats
	net_stats = ganist.step(None, None, stats_only=True)

	### fid
	fid = eval_fid(ganist.sess, r_samples, g_samples)

	return fid, fid, net_stats


if __name__ == '__main__':
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
	svhn_test_path = '/media/evl/Public/Mahyar/Data/svhn/test/svhn_ndarray.cpk'
	svhn_train_path = '/media/evl/Public/Mahyar/Data/svhn/train/svhn_ndarray.cpk'
	train_bbox, train_im, train_names = read_svhn_prep(svhn_train_path)
	test_bbox, test_im, test_names = read_svhn_prep(svhn_test_path)

	### svhn_cut (as content images)
	svhn_32_train = '/media/evl/Public/Mahyar/Data/svhn/train_32x32.mat'
	svhn_32_test = '/media/evl/Public/Mahyar/Data/svhn/test_32x32.mat'
	train_co, train_co_labs = read_svhn_32(svhn_32_train)
	test_co, test_co_labs = read_svhn_32(svhn_32_test)
	
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
		print >>fs, '>>> g_vars: %d --- d_vars: %d --- e_vars: %d' \
			% (condet.g_vars_count, condet.d_vars_count, condet.e_vars_count)
	
	'''
	GAN SETUP SECTION
	'''
	### train condet
	train_condet(condet, train_co, train_im)

	### load ganist **g_num**
	# condet.load(condet_path % run_seed)
	### gset draws: run sample_draw before block_draw_top to load learned gset prior
	#gset_sample_draw(ganist, 10)
	gset_block_draw(ganist, 10, log_path+'/gset_samples.png', border=True)
	gset_block_draw_top(ganist, 10, log_path+'/gset_top_samples.png', pr_th=0.99 / ganist.g_num)
	#sys.exit(0)

	'''
	GAN DATA EVAL
	'''
	gan_model = ganist#vae
	sampler = ganist.step#vae.step
	### sample gen data and draw **mt**
	g_samples = sample_ganist(gan_model, sample_size, sampler=sampler,
		z_im=r_samples[0:sample_size, ...])
	#im_block_draw(g_samples, 10, log_path_draw+'/gen_samples.png')
	im_block_draw(r_samples, 5, log_path_draw+'/real_samples.png', border=True)
	im_block_draw(g_samples, 5, log_path_draw+'/gen_samples.png', border=True)
	#sys.exit(0)

	sess.close()

