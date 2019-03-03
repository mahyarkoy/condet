import numpy as np
import tensorflow as tf
import os
import spatial_transformer as stn
#os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID" # so the IDs match nvidia-smi
#os.environ["CUDA_VISIBLE_DEVICES"] = "1" # "0, 1" for multiple

#np.random.seed(13)
#tf.set_random_seed(13)

tf_dtype = tf.float32
np_dtype = 'float32'

### Operations
def lrelu(x, leak=0.2, name="lrelu"):
	return tf.nn.leaky_relu(x, leak, name)

def conv2d(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope="conv2d", reuse=False, 
		   padding='same', use_bias=True, trainable=True):
	
	k_init = tf.truncated_normal_initializer(stddev=0.02)
	conv = tf.layers.conv2d(
		input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
		padding=padding, use_bias=use_bias, 
		kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)

	return conv

def conv2d_tr(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope="conv2d_tr", reuse=False, 
		   padding='same', use_bias=True, trainable=True):
    
    k_init = tf.truncated_normal_initializer(stddev=0.02)
    conv_tr = tf.layers.conv2d_transpose(
            input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
            padding=padding, use_bias=use_bias, 
            kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)
    
    return conv_tr

def dense_batch(x, h_size, scope, phase, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
	with tf.variable_scope(scope):
		h2 = tf.contrib.layers.batch_norm(h1, center=True, scale=True, is_training=phase, scope='bn_'+str(reuse))
	return h2

def dense(x, h_size, scope, reuse=False):
	with tf.variable_scope(scope, reuse=reuse):
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
		h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.contrib.layers.xavier_initializer(uniform=True))
	return h1

### GAN Class definition
class Condet:
	def __init__(self, sess, log_dir='logs'):
		### run parameters
		self.log_dir = log_dir
		self.sess = sess

		### optimization parameters
		self.g_lr = 2e-4
		self.g_beta1 = 0.9
		self.g_beta2 = 0.999
		self.d_lr = 2e-4
		self.d_beta1 = 0.9
		self.d_beta2 = 0.999
		self.e_lr = 2e-4
		self.e_beta1 = 0.9
		self.e_beta2 = 0.999

		### loss weights
		self.gp_loss_weight = 10.0
		self.rec_loss_weight = 0.0
		self.g_init_loss_weight = 0.0
		self.g_gp_loss_weight = 0.0
		self.enc_loss_weight = 1.0

		self.im_dg_loss_weight = 0.0
		self.use_gen = False

		self.stn_init_loss_weight = 10.0
		self.stn_boundary_loss_weight = 10.0
		self.stn_scale_loss_weight = 1000.0

		### network parameters
		self.stn_num = 10
		self.z_dim = 100
		self.z_range = 1.0
		self.data_dim = [128, 128, 3]
		self.co_dim = [64, 64, 3]
		self.stn_size = self.co_dim[:2] ### co_dim
	
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		#self.d_act = tf.tanh
		#self.g_act = tf.tanh
		self.d_act = lrelu
		self.g_act = tf.nn.relu
		self.rec_act = lrelu
		self.a_act = lrelu
		self.s_act = lrelu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		with tf.name_scope('condet'):
			### define placeholders for image and content inputs
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			self.co_input = tf.placeholder(tf_dtype, [None]+self.co_dim, name='co_input')
			self.z_input = tf.placeholder(tf.int32, [None], name='z_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')
			self.run_count = tf.placeholder(tf_dtype, name='run_count')
			self.dscore_mean_input = tf.placeholder(tf_dtype, name='dscore_mean_input')
			self.dscore_std_input = tf.placeholder(tf_dtype, name='dscore_std_input')
			self.penalty_weight = tf.pow(0.9, self.run_count)
			self.g_init_penalty_weight = tf.pow(0.9, self.run_count)

			### build generators (encoder)
			#self.g_layer = self.build_gen(self.im_input, self.train_phase)
			#self.g_layer = self.im_input

			### d_mean and d_std updates
			self.dscore_mean = tf.get_variable('dscore_mean', dtype=tf_dtype, initializer=0.0)
			self.dscore_std = tf.get_variable('dscore_std', dtype=tf_dtype, initializer=0.0)
			self.dscore_mean_opt = tf.assign(self.dscore_mean, self.dscore_mean_input)
			self.dscore_std_opt = tf.assign(self.dscore_std, self.dscore_std_input)

			### theta decay updates
			self.theta_decay = tf.get_variable('theta_decay', dtype=tf_dtype, initializer=.0)
			self.theta_decay_opt = tf.assign(self.theta_decay, self.theta_decay * 0.999)

			### build stn
			self.stn_layer, self.theta = self.build_stn(self.im_input, self.z_input, self.train_phase)
			print '>>> STN shape: ', self.stn_layer.get_shape().as_list()

			### theta encoder loss
			self.enc_logits = self.build_enc(self.theta, self.train_phase)
			stn_labels = tf.tile(tf.reshape(np.arange(self.stn_num), [1, self.stn_num]),
					[tf.shape(self.im_input)[0], 1])
			self.enc_loss = tf.reduce_mean(
				tf.nn.softmax_cross_entropy_with_logits(
					labels=tf.one_hot(tf.reshape(stn_labels, [-1]), self.stn_num, 
					dtype=tf_dtype), logits=self.enc_logits))

			### stn init penalty
			self.stn_init_loss = tf.reduce_mean(
				tf.square(self.theta[:,0] - 1.0) + tf.square(self.theta[:,2]) + \
				tf.square(self.theta[:,4] - 1.0) + tf.square(self.theta[:,5]))
			
			### stn boundary penalty
			bbox_l = -self.theta[:,0] + self.theta[:, 2]
			bbox_t = -self.theta[:,4] + self.theta[:, 5]
			bbox_r = self.theta[:,0] + self.theta[:, 2]
			bbox_b = self.theta[:,4] + self.theta[:, 5]
			self.stn_boundary_loss = tf.reduce_mean(
				-tf.minimum(bbox_l, -1.0) - tf.minimum(bbox_t, -1.0) + \
				tf.maximum(bbox_r, 1.0) + tf.maximum(bbox_b, 1.0))

			### stn scale penalty
			self.stn_scale_loss = -tf.reduce_mean(
				tf.minimum(self.theta[:,0], 0.2) + tf.minimum(self.theta[:,4], 0.2))

			### build generators (encoders)
			self.im_g_layer = self.build_gen(self.stn_layer, self.train_phase, 'im_gen')
			#self.co_g_layer = self.build_gen(self.co_input, self.train_phase, 'co_gen', share=True)

			### use reconstructors (decoders)
			self.im_rec_layer = self.build_gen(self.im_g_layer, self.train_phase, 'co_gen', share=True, reuse=True)
			#self.co_rec_layer = self.build_gen(self.co_g_layer, self.train_phase, 'im_gen', share=True, reuse=True)

			### build discriminator (critic)
			self.co_r_logits, self.co_r_hidden = self.build_dis(self.co_input, self.train_phase, 'co_dis')
			self.co_g_logits, self.co_g_hidden = self.build_dis(self.im_g_layer, self.train_phase, 'co_dis', reuse=True)
			#self.im_r_logits, self.im_r_hidden = self.build_dis(tf.stop_gradient(self.stn_layer), self.train_phase, 'im_dis')
			#self.im_g_logits, self.im_g_hidden = self.build_dis(self.co_g_layer, self.train_phase, 'im_dis', reuse=True)

			self.co_g_logits_re = tf.reshape(self.co_g_logits, [-1, self.stn_num])
			### build attention over multiple stns
			#self.g_multi_att = self.build_multi_att(self.co_g_hidden, self.train_phase)
			### use greedy attention over multiple stns
			#co_g_max = tf.reduce_max(self.co_g_logits_re, axis=1, keepdims=True)
			#self.g_multi_att = tf.stop_gradient(
			#	tf.cast(tf.equal(self.co_g_logits_re, co_g_max), tf_dtype))
			### use average attention
			self.g_multi_att = tf.ones_like(self.co_g_logits_re, dtype=tf_dtype) / self.stn_num

			### build batch attention, shape: (B, k, k, 1)
			#self.r_att = self.build_att(self.r_hidden, self.train_phase)
			#self.g_att = self.build_att(self.g_hidden, self.train_phase, reuse=True)
			##self.r_att = tf.ones_like(self.r_att) / tf.cast(tf.size(self.r_att) / tf.shape(self.r_att)[0], tf_dtype)
			##self.g_att = tf.ones_like(self.g_att) / tf.cast(tf.size(self.g_att) / tf.shape(self.g_att)[0], tf_dtype)
			#print '>>> r_att shape: ', self.r_att.get_shape().as_list()
			#print '>>> g_att shape: ', self.g_att.get_shape().as_list()

			### build real attention ground truth (center 1 hot)
			#inds = tf.shape(self.r_att) / 2
			#r_att_shape = tf.shape(self.r_att)
			#updates = tf.constant([1.0])
			#r_att_gt = tf.scatter_nd([inds[1:3]], updates, r_att_shape[1:3])
			#r_att_gt = tf.reshape(r_att_gt, [1, r_att_shape[1], r_att_shape[2], 1])
			#print '>>> r_att_gt shape: ', r_att_gt.get_shape().as_list()
			### build real attention loss
			#self.r_att_loss = tf.reduce_mean(tf.reduce_sum(tf.square(r_att_gt - self.r_att), axis=[1,2,3]))

			### debug g_att
			### build real attention ground truth (center 1 hot)
			##inds = tf.shape(self.g_att) - 1
			##g_att_shape = tf.shape(self.g_att)
			#inds = tf.constant([1, 1, 7, 1])
			#g_att_shape = tf.constant([1, 8, 8, 1])
			#updates = tf.constant([1.0])
			#g_att_gt = tf.scatter_nd([inds[1:3]], updates, g_att_shape[1:3])
			#g_att_gt = tf.reshape(g_att_gt, [1, g_att_shape[1], g_att_shape[2], 1])
			#print '>>> g_att_gt shape: ', g_att_gt.get_shape().as_list()
			##self.g_att = g_att_gt

			### random vars for real gen manifold interpolation
			int_rand = tf.random_uniform([tf.shape(self.im_input)[0]], minval=0.0, maxval=1.0, dtype=tf_dtype)
			int_rand = tf.reshape(int_rand, [-1, 1, 1, 1])
			idx_rand = tf.random_uniform([tf.shape(self.im_input)[0]], minval=0, maxval=self.stn_num, dtype=tf.int32)
			### select one stn for each image: im_g_layer_sub shape (B, H, W, C)
			z_1hot_stn = tf.reshape(tf.one_hot(idx_rand, self.stn_num, dtype=tf_dtype), [-1, self.stn_num, 1, 1, 1])
			z_map_stn = tf.tile(z_1hot_stn, [1, 1]+self.co_dim)
			im_g_layer_sub = tf.reduce_sum(tf.reshape(self.im_g_layer, [-1, self.stn_num]+self.co_dim) * z_map_stn, axis=1)
			### real gen manifold interpolation
			co_rg_layer = (1.0 - int_rand) * im_g_layer_sub + int_rand * self.co_input
			co_rg_logits, _ = self.build_dis(co_rg_layer, self.train_phase, 'co_dis', reuse=True)
			#im_rg_layer = (1.0 - int_rand) * self.co_g_layer + int_rand * self.stn_layer
			#im_rg_logits, _ = self.build_dis(im_rg_layer, self.train_phase, 'im_dis', reuse=True)

			### build d losses
			self.co_d_r_loss, self.co_d_g_loss, self.co_gp_loss, self.co_grad_norm = \
				self.build_dis_loss(self.co_r_logits, self.co_g_logits, co_rg_logits, co_rg_layer)

			#self.im_d_r_loss, self.im_d_g_loss, self.im_gp_loss, self.im_grad_norm = \
			#	self.build_dis_loss(self.im_r_logits, self.im_g_logits, im_rg_logits, im_rg_layer)

			### d loss mean simple (no batch)
			self.co_d_loss_mean = \
				tf.reduce_mean(self.co_d_r_loss + \
					tf.reduce_sum(self.g_multi_att * \
						tf.reshape(self.co_d_g_loss, [-1, self.stn_num]), axis=1, keepdims=True)) + \
				self.gp_loss_weight * tf.reduce_mean(self.co_gp_loss)

			#self.im_d_loss_mean = tf.reduce_mean(self.im_d_r_loss + self.im_d_g_loss) + \
			#	self.gp_loss_weight * tf.reduce_mean(self.im_gp_loss)

			self.d_loss_total = self.co_d_loss_mean #+ self.im_dg_loss_weight * self.im_d_loss_mean

			### build g loss and rec losses
			self.co_g_loss, self.im_rec_loss, self.im_g_init_loss, self.co_g_gp_loss = \
				self.build_gen_loss(self.co_g_logits, 
					self.stn_layer, self.im_g_layer, self.im_rec_layer, self.im_input, self.theta)
			#self.im_g_loss, self.co_rec_loss, self.co_g_init_loss, _ = self.build_gen_loss(self.im_g_logits, 
			#	self.co_input, self.co_g_layer, self.co_rec_layer)

			### g loss mean simple (no batch)
			self.co_g_loss_mean = \
				tf.reduce_mean(tf.reduce_sum(self.g_multi_att * \
					tf.reshape(self.co_g_loss, [-1, self.stn_num]), axis=1))
			#self.im_g_loss_mean = tf.reduce_mean(self.im_g_loss)

			### g_att and grad logs
			#g_att_grad = tf.gradients(self.g_loss_mean, self.g_att)
			#print '>>> g_att_grad shape: ', g_att_grad[0].get_shape().as_list()
			#self.g_att_grad_mean = tf.reduce_mean(g_att_grad[0], axis=0)
			#self.g_loss_mean_sep = tf.reduce_mean(self.g_loss, axis=0)
			#self.d_r_loss_mean_sep = tf.reduce_mean(self.d_r_loss, axis=0)

			### reconstruction loss mean **weighted** (upsampling with deconv: d_h=2^(nd), k(t+1) = 2k(t)-1 + k(1)-1)
			#self.g_att_us = tf.image.resize_nearest_neighbor(self.g_att, tf.shape(self.i_layer)[1:3])
			#k_init = tf.constant_initializer(1.0)
			#self.g_att_us = conv2d_tr(self.g_att, 1, k_h=29, k_w=29, d_h=8, d_w=8, scope='rec_deconv_g', k_init=k_init, trainable=False)
			#self.r_att_us = conv2d_tr(self.r_att, 1, k_h=29, k_w=29, d_h=8, d_w=8, scope='rec_deconv_r', k_init=k_init, trainable=False)
			#self.rec_loss_mean = tf.reduce_mean(tf.reduce_sum(
			#		self.g_att_us * tf.square(self.i_layer - self.im_input), axis=[1,2,3]))

			##self.g_grad_norm = tf.norm(tf.reshape(
			##	tf.gradients(self.g_loss, self.g_layer), [-1, np.prod(self.co_dim)]), axis=1)

			### g loss combination
			self.g_loss_total = self.co_g_loss_mean + \
				self.penalty_weight * self.stn_init_loss_weight * self.stn_init_loss + \
				self.stn_boundary_loss_weight * self.stn_boundary_loss + \
				self.stn_scale_loss_weight * self.stn_scale_loss #+ \
				#self.enc_loss_weight * self.enc_loss
				#self.im_dg_loss_weight * self.im_g_loss_mean
				#self.rec_loss_weight * (self.co_rec_loss + self.im_rec_loss)
				#self.g_init_penalty_weight * self.g_init_loss_weight * (self.co_g_init_loss + self.im_g_init_loss)
				#self.g_gp_loss_weight * tf.reduce_mean(self.co_g_gp_loss)

			### collect params
			self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
			self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
			self.a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a_net")
			self.i_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "i_net")
			self.e_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "e_net")
			self.s_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "s_net")
			self.s_vars_bn = list()
			for v in tf.global_variables():
				if 's_net' in v.name and 'bn' in v.name and 'beta' not in v.name:
					self.s_vars_bn.append(v)
			#print '>>> S_VARS_BN: ', self.s_vars_bn

			### compute stat of weights
			self.nan_vars = 0.
			self.inf_vars = 0.
			self.zero_vars = 0.
			self.big_vars = 0.
			self.count_vars = 0
			for v in self.g_vars + self.d_vars + self.a_vars + self.i_vars + self.s_vars:
				self.nan_vars += tf.reduce_sum(tf.cast(tf.is_nan(v), tf_dtype))
				self.inf_vars += tf.reduce_sum(tf.cast(tf.is_inf(v), tf_dtype))
				self.zero_vars += tf.reduce_sum(tf.cast(tf.square(v) < 1e-6, tf_dtype))
				self.big_vars += tf.reduce_sum(tf.cast(tf.square(v) > 1., tf_dtype))
				self.count_vars += tf.reduce_prod(v.get_shape())
			self.count_vars = tf.cast(self.count_vars, tf_dtype)
			#self.nan_vars /= self.count_vars 
			#self.inf_vars /= self.count_vars
			self.zero_vars /= self.count_vars
			self.big_vars /= self.count_vars

			self.g_vars_count = 0
			self.d_vars_count = 0
			self.a_vars_count = 0
			self.i_vars_count = 0
			self.s_vars_count = 0
			for v in self.g_vars:
				self.g_vars_count += int(np.prod(v.get_shape()))
			for v in self.d_vars:
				self.d_vars_count += int(np.prod(v.get_shape()))
			for v in self.a_vars:
				self.a_vars_count += int(np.prod(v.get_shape()))
			for v in self.i_vars:
				self.i_vars_count += int(np.prod(v.get_shape()))
			for v in self.s_vars:
				self.s_vars_count += int(np.prod(v.get_shape()))

			### build optimizers
			self.update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print '>>> update_ops list: ', self.update_ops
			with tf.control_dependencies(self.update_ops):
				self.d_opt = tf.train.AdamOptimizer(
					self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
					self.d_loss_total, var_list=self.d_vars)

				with tf.control_dependencies([self.theta_decay_opt]):
					self.g_opt = tf.train.AdamOptimizer(
						self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
						self.g_loss_total, var_list=self.g_vars+self.s_vars+self.a_vars)

				self.e_opt = tf.train.AdamOptimizer(
					self.e_lr, beta1=self.e_beta1, beta2=self.e_beta2).minimize(
					self.enc_loss, var_list=self.e_vars)

				#self.a_opt = tf.train.AdamOptimizer(
				#	self.a_lr, beta1=self.a_beta1, beta2=self.a_beta2).minimize(
				#	self.g_loss_total, var_list=self.a_vars)
				#self.i_opt = tf.train.AdamOptimizer(
				#	self.i_lr, beta1=self.i_beta1, beta2=self.i_beta2).minimize(
				#	self.rec_loss, var_list=self.i_vars)

			### theta summaries
			#sh_sum = tf.summary.histogram("scale_h_stn", self.theta_stn[:, 0])
			#th_sum = tf.summary.histogram("trans_h_stn", self.theta_stn[:, 2])
			#sw_sum = tf.summary.histogram("scale_w_stn", self.theta_stn[:, 4])
			#tw_sum = tf.summary.histogram("trans_w_stn", self.theta_stn[:, 5])

			### loss summaries **g_num**
			co_g_loss_sum = tf.summary.scalar("co_g_loss", self.co_g_loss_mean)
			co_d_loss_sum = tf.summary.scalar("co_d_loss", self.co_d_loss_mean)
			#im_g_loss_sum = tf.summary.scalar("im_g_loss", self.im_g_loss_mean)
			#im_d_loss_sum = tf.summary.scalar("im_d_loss", self.im_d_loss_mean)
			#co_rec_loss_sum = tf.summary.scalar("co_rec_loss", self.co_rec_loss)
			#im_rec_loss_sum = tf.summary.scalar("im_rec_loss", self.im_rec_loss)
			g_loss_sum = tf.summary.scalar("g_loss_total", self.g_loss_total)
			d_loss_sum = tf.summary.scalar("d_loss_total", self.d_loss_total)
			e_loss_sum = tf.summary.scalar("enc_loss", self.enc_loss)
			self.summary = tf.summary.merge(
				[g_loss_sum, d_loss_sum, 
				co_g_loss_sum, co_d_loss_sum,
				e_loss_sum]) 
				#im_g_loss_sum, im_d_loss_sum, co_rec_loss_sum, im_rec_loss_sum])
				#sh_sum, th_sum, sw_sum, tw_sum])

	def build_dis_loss(self, r_logits, g_logits, rg_logits, rg_layer):
		### build d losses
		if self.d_loss_type == 'log':
			d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=r_logits, labels=tf.ones_like(r_logits, tf_dtype))
			d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=g_logits, labels=tf.zeros_like(g_logits, tf_dtype))
			d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=rg_logits, labels=tf.ones_like(rg_logits, tf_dtype))
		elif self.d_loss_type == 'was':
			d_r_loss = -r_logits 
			d_g_loss = g_logits
			d_rg_loss = -rg_logits
		else:
			raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)

		### gradient penalty
		### NaN free norm gradient
		rg_grad = tf.gradients(rg_logits, rg_layer)
		rg_grad_flat = tf.contrib.layers.flatten(rg_grad[0])
		rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 1.
		rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
		#rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
		rg_grad_abs =  0. * rg_grad_flat
		rg_grad_norm = tf.where(rg_grad_ok, 
			tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
		gp_loss = tf.square(rg_grad_norm - 1.0)
		### for logging
		rg_grad_norm_output = tf.norm(rg_grad_flat, axis=1)

		return d_r_loss, d_g_loss, gp_loss, rg_grad_norm_output

	def build_gen_loss(self, g_logits, data_layer, g_layer, rec_layer, im_input=None, theta=None):
		if self.g_loss_type == 'log':
				g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
					logits=g_logits, labels=tf.zeros_like(g_logits, tf_dtype))
		elif self.g_loss_type == 'mod':
			g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
				logits=g_logits, labels=tf.ones_like(g_logits, tf_dtype))
		elif self.g_loss_type == 'was':
			g_loss = -g_logits
		else:
			raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)
		
		rec_loss = tf.reduce_mean(
			tf.reduce_sum(tf.square(data_layer - rec_layer), axis=[1,2,3]))

		init_loss = tf.reduce_mean(
			tf.reduce_sum(tf.square(tf.stop_gradient(data_layer) - g_layer), axis=[1,2,3]))

		gp_loss = 0.0
		if im_input is not None:
			for t in [0, 2, 4, 5]:	
				### gradient penalty
				### NaN free norm gradient
				rg_grad = tf.gradients(theta[:,t], im_input)
				rg_grad_flat = tf.contrib.layers.flatten(rg_grad[0])
				rg_sum_sq = tf.reduce_sum(tf.square(rg_grad_flat), axis=1)
				rg_grad_ok = rg_sum_sq > 0.
				rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
				rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
				#rg_grad_abs =  0. * rg_grad_flat
				rg_grad_norm = tf.where(rg_grad_ok, 
					tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
				gp_loss = gp_loss + tf.square(tf.minimum(rg_grad_norm, 1.) - 1.)

		return g_loss, rec_loss, init_loss, gp_loss

	def build_gen(self, z, train_phase, scope, reuse=False, share=False):
		if self.use_gen is False:
			return z
		act = self.g_act
		bn = tf.contrib.layers.batch_norm
		im_size = self.co_dim[0]
		batch_size = tf.shape(z)[0]
		### shared
		with tf.variable_scope('g_net'):
			#with tf.variable_scope('shared', reuse=share):
				#h1 = act(bn(conv2d(z, 64, d_h=2, d_w=2, scope='conv1'), is_training=True))
				#h2 = act(bn(conv2d(h1, 32, d_h=2, d_w=2, scope='conv2'), is_training=True))

			with tf.variable_scope(scope, reuse=reuse):
				#h1_tr = act(bn(conv2d_tr(h2, 64, d_h=2, d_w=2, scope='conv1_tr'), is_training=train_phase))
				#h2_tr = conv2d_tr(h1_tr, self.co_dim[-1], d_h=2, d_w=2, scope='conv2_tr')
				#o = tf.tanh(h2_tr)

				### add some noise
				z_n_dim = 1
				z_n = tf.random_uniform([tf.shape(z)[0], tf.shape(z)[1], tf.shape(z)[2], z_n_dim], minval=-1.0, maxval=1.0, dtype=tf_dtype)
				#z_n = tf.reshape(z_n, [tf.shape(z)[0], 1, 1, z_n_dim])
				#z_n = tf.tile(z_n, [1, tf.shape(z)[1], tf.shape(z)[2], 1])
				zc = tf.concat([z, z_n], axis=-1)
				### transform
				h1 = act(bn(conv2d(zc, 32, d_h=1, d_w=1, scope='conv1'), is_training=True))
				#h2 = act(bn(conv2d(h1, 32, d_h=1, d_w=1, scope='conv2'), is_training=True))
				h3 = conv2d(h1, self.co_dim[-1], d_h=1, d_w=1, scope='conv3')
				o = tf.tanh(h3)
		return o

	def build_rec(self, x, train_phase):
		act = self.rec_act
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('i_net'):
			h1 = act(conv2d(x, 32, scope='conv1'))
			h2 = act(bn(conv2d(h1, 32, scope='conv2'), is_training=train_phase))
			h3 = conv2d(h2, 3, scope='conv3')
			o = tf.tanh(h3)
		return o

	def build_dis(self, data_layer, train_phase, scope, reuse=False):
		act = self.d_act
		bn = tf.contrib.layers.batch_norm
		ln = tf.contrib.layers.layer_norm
		with tf.variable_scope('d_net'):
			with tf.variable_scope(scope):
				### encoding the 64*64*3 image with conv into 8*8*1
				h1 = act(conv2d(data_layer, 32, d_h=2, d_w=2, scope='conv1', reuse=reuse))
				h2 = act(conv2d(h1, 64, d_h=2, d_w=2, scope='conv2', reuse=reuse))
				h3 = act(conv2d(h2, 128, d_h=2, d_w=2, scope='conv3', reuse=reuse))
				flat = tf.contrib.layers.flatten(h3)
				o = dense(flat, 1, 'fco', reuse=reuse)
				#o = conv2d(h3, 1, k_h=1, k_w=1, scope='conv4', reuse=reuse)
		return o, h3

	def build_multi_att(self, hidden_layer, train_phase, reuse=False):
		act = self.d_act
		with tf.variable_scope('a_net'):
			flat = tf.contrib.layers.flatten(tf.stop_gradient(hidden_layer))
			o = dense(flat, 1, 'multi_att_fc')
			o_re = tf.reshape(o, [-1, self.stn_num])
			att = tf.nn.softmax(o_re)
		return tf.nn.softmax(o_re)

	def build_enc(self, theta_layer, train_phase, reuse=False):
		act = self.d_act
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('e_net'):
			loc = tf.stack([theta_layer[:, 2], theta_layer[:, 5]], axis=1)
			h1 = act(bn(dense(loc, 16, 'ehfc', reuse=reuse), 
					reuse=reuse, is_training=train_phase, scope='bn'))
			o = dense(h1, self.stn_num, 'efc', reuse=reuse)
		return o

	def build_stn(self, data_layer, z_data, train_phase, reuse=False):
		#scale, trans = fc(6)
		act = self.s_act
		bn = tf.contrib.layers.batch_norm
		stn_list = list()
		theta_list = list()
		theta_init_list = list()
		theta_stn_list = list()
		with tf.variable_scope('s_net'):
			### theta net
			### shared layers go here
			#h1 = act(conv2d(data_layer, 32, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			#h2 = act(bn(conv2d(h1, 64, d_h=2, d_w=2, scope='conv2', reuse=reuse), 
			#	reuse=reuse, scope='bn2', is_training=train_phase))
			#h3 = act(bn(conv2d(h2, 128, d_h=2, d_w=2, scope='conv3', reuse=reuse), 
			#	reuse=reuse, scope='bn3', is_training=train_phase))
			#flat = tf.contrib.layers.flatten(h3)
			for si in range(self.stn_num):
				with tf.variable_scope('snum_%d' % si):
					### theta net
					h1 = act(conv2d(data_layer, 32, d_h=2, d_w=2, scope='conv1', reuse=reuse))
					h2 = act(bn(conv2d(h1, 64, d_h=2, d_w=2, scope='conv2', reuse=reuse), 
						reuse=reuse, scope='bn2', is_training=train_phase))
					h3 = act(bn(conv2d(h2, 128, d_h=2, d_w=2, scope='conv3', reuse=reuse), 
						reuse=reuse, scope='bn3', is_training=train_phase))
					flat = tf.contrib.layers.flatten(h3)
					sh = tf.sigmoid(dense(flat, 1, 'hscale', reuse=reuse))
					sw = tf.sigmoid(dense(flat, 1, 'wscales', reuse=reuse))
					th = 2.0*tf.sigmoid(dense(flat, 1, 'htrans', reuse=reuse))
					tw = 2.0*tf.sigmoid(dense(flat, 1, 'wtrans', reuse=reuse))

					### biases
					sh_b = tf.get_variable('hscale_bias', dtype=tf_dtype, 
						initializer=1.0)
					sw_b = tf.get_variable('wscale_bias', dtype=tf_dtype, 
						initializer=1.0)
					th_b = tf.get_variable('htrans_bias', 
						dtype=tf_dtype, initializer=-1.0+np.random.uniform(-0.5, 0.5))
					tw_b = tf.get_variable('wtrans_bias', 
						dtype=tf_dtype, initializer=-1.0+np.random.uniform(-0.5, 0.5))

					### theta init setup
					z = tf.zeros([tf.shape(data_layer)[0], 1], dtype=tf_dtype)
					#theta_init = tf.get_variable('theta_init', initializer=tf.constant([[1., 0., 0., 0., 1., 0.]]))
					#theta_init = tf.constant([[1., 0., 0., 0., 1., 0.]], dtype=tf_dtype)
					theta_init = tf.reshape(tf.stack([sh_b, 0.0, th_b, 0.0, sw_b, tw_b], axis=0), [1, 6])
					#print '>>> Theta Init shape: ', theta_init.get_shape().as_list()
					#theta = (1.0-self.theta_decay) * tf.concat([sh, z, th, z, sw, tw], axis=1) + self.theta_decay * theta_init
					theta_stn = tf.concat([-sh, z, th, z, -sw, tw], axis=1)
					theta = theta_stn + theta_init
					#print '>>> Theta shape: ', theta.get_shape().as_list()

					### stn net
					stn_layer = tf.reshape(stn.transformer(data_layer, theta, self.stn_size), [-1]+self.co_dim)

					### collect
					stn_list.append(stn_layer)
					theta_list.append(theta)
					theta_init_list.append(theta_init)
					theta_stn_list.append(theta_stn)

			### choose the theta and stn based on z input
			#z_1hot_stn = tf.reshape(tf.one_hot(z_data, self.stn_num, dtype=tf_dtype), [-1, self.stn_num, 1, 1, 1])
			#z_map_stn = tf.tile(z_1hot_stn, [1, 1]+self.co_dim)
			#z_1hot_theta = tf.reshape(tf.one_hot(z_data, self.stn_num, dtype=tf_dtype), [-1, self.stn_num, 1])
			#z_map_theta = tf.tile(z_1hot_theta, [1, 1, 6])
			#stn_o = tf.reduce_sum(tf.stack(stn_list, axis=1) * z_map_stn, axis=1)
			#theta_o = tf.reduce_sum(tf.stack(theta_list, axis=1) * z_map_theta, axis=1)

			### stack stn and theta: (B*stn_num, H, W, C) and (B*stn_num, 6)
			stn_o = tf.reshape(tf.stack(stn_list, axis=1), [-1,]+self.co_dim)
			theta_o = tf.reshape(tf.stack(theta_list, axis=1), [-1, 6])
			self.theta_stn = tf.reshape(tf.stack(theta_stn_list, axis=1), [-1, 6])
			self.theta_init = tf.reshape(tf.stack(theta_init_list, axis=1), [-1, 6])
			return stn_o, theta_o

	def start_session(self):
		self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)
		### load only
		self.saver_var_only = tf.train.Saver(
			self.s_vars_bn + self.s_vars + self.d_vars) #+self.a_vars+[self.dscore_mean, self.dscore_std])
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver_var_only.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	'''
	if vals_list is none: read the constant vars (dscore_mean dscore_std)
	otherwise: update the constant vars with the values in vals_list
	'''
	def update_const_vars(self, vals_list=None):
		if vals_list is None:
			res_list = [self.dscore_mean, self.dscore_std]
			res_list = self.sess.run(res_list)
		else:
			feed_dict = {self.dscore_mean_input: vals_list[0], self.dscore_std_input: vals_list[1]}
			res_list = [self.dscore_mean_opt, self.dscore_std_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		return res_list

	def step(self, im_data, co_data=None, gen_update=False, 
		gen_only=False, disc_only=False, stats_only=False, z_data=None, 
		att_only_co=False, att_only_im=False, run_count=0.0):

		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		batch_size = im_data.shape[0]		
		im_data = im_data.astype(np_dtype) if im_data is not None else None
		co_data = co_data.astype(np_dtype) if co_data is not None else None

		if z_data is None:
			#g_th = self.g_num
			#z_pr = np.exp(self.pg_temp * self.g_rl_pvals[:g_th])
			#z_pr = z_pr / np.sum(z_pr)
			#z_data = np.random.choice(g_th, size=batch_size, p=z_pr)
			z_data = np.random.randint(low=0, high=self.stn_num, size=batch_size)

		### only forward attention on im_data using co_input (no transformation)
		#if att_only_co:
		#	feed_dict = {self.co_input: im_data, self.train_phase: False}
		#	res_list = self.sess.run(self.r_att_us, feed_dict=feed_dict)
		#	return res_list

		### only forward attention on im_data using im_input (with transformation)
		#if att_only_im:
		#	feed_dict = {self.im_input: im_data, self.train_phase: False}
		#	res_list = [self.g_layer, self.i_layer, self.g_att_us]
		#	res_list = self.sess.run(res_list, feed_dict=feed_dict)
		#	return res_list

		### only forward stn on im_data using im_input
		if att_only_im:
			feed_dict = {self.im_input: im_data, self.z_input: z_data, self.train_phase: False}
			res_list = [self.stn_layer, self.co_g_logits, self.theta, self.theta_stn, self.theta_init]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list

		### only forward generator on co_data using im_input
		if att_only_co:
			feed_dict = {self.co_input: im_data, self.train_phase: False}
			res_list = [self.co_r_logits]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			return res_list[0]

		### only forward generator on im_data
		if gen_only:
			feed_dict = {self.im_input: im_data, self.z_input: z_data, self.train_phase: False}
			g_layer = self.sess.run(self.im_g_layer, feed_dict=feed_dict)
			return g_layer

		### only forward discriminator to compute norms
		if disc_only:
			feed_dict = {self.co_input: co_data, self.im_input: im_data,  self.z_input: z_data, self.train_phase: False}
			#res_list = [self.rg_grad_norm_output, self.d_r_loss_mean_sep, self.g_loss_mean_sep]
			res_list = [self.co_grad_norm]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
			#return res_list[0].flatten(), res_list[1].flatten(), res_list[2].flatten()
			return res_list[0].flatten(), 0., 0.

		### run one training step on discriminator, otherwise on generator, and log **g_num**
		feed_dict = {self.co_input: co_data, self.im_input: im_data, self.z_input: z_data, 
					self.run_count: run_count, self.train_phase: True}
		if not gen_update:
			res_list = [self.im_g_layer, self.summary, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			### encoder opt
			#self.sess.run(self.e_opt, feed_dict=feed_dict)
			### stn opt
			res_list = [self.im_g_layer, self.summary, self.g_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		
		### return summary and g_layer
		return res_list[1], res_list[0]
