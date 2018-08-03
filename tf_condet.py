import numpy as np
import tensorflow as tf
import os
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
	
	conv = tf.layers.conv2d(
		input_, output_dim, [k_h, k_w], strides=[d_h, d_w], 
		padding=padding, use_bias=use_bias, 
		kernel_initializer=k_init, name=scope, reuse=reuse, trainable=trainable)

	return conv

def conv2d_tr(input_, output_dim,
		   k_h=5, k_w=5, d_h=1, d_w=1, k_init=tf.contrib.layers.xavier_initializer(),
		   scope="conv2d_tr", reuse=False, 
		   padding='same', use_bias=True, trainable=True):
    
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
		h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense')
		#h1 = tf.contrib.layers.fully_connected(x, h_size, activation_fn=None, scope='dense', weights_initializer=tf.truncated_normal_initializer(stddev=0.02))
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
		self.g_beta1 = 0.5
		self.g_beta2 = 0.5
		self.d_lr = 2e-4
		self.d_beta1 = 0.5
		self.d_beta2 = 0.5
		self.e_lr = 2e-4
		self.e_beta1 = 0.9
		self.e_beta2 = 0.999
		self.pg_lr = 1e-3
		self.pg_beta1 = 0.5
		self.pg_beta2 = 0.5

		### network parameters **g_num** **mt**
		### >>> dataset sensitive: data_dim
		self.z_dim = 100
		self.z_range = 1.0
		self.data_dim = [None, None, 3]
		self.gp_loss_weight = 10.0
		self.rec_loss_weight = 1.0
	
		self.d_loss_type = 'was'
		self.g_loss_type = 'was'
		#self.d_act = tf.tanh
		#self.g_act = tf.tanh
		self.d_act = lrelu
		self.g_act = tf.nn.relu

		### init graph and session
		self.build_graph()
		self.start_session()

	def build_graph(self):
		with tf.name_scope('condet'):
			### define placeholders for image and content inputs
			self.im_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='im_input')
			self.co_input = tf.placeholder(tf_dtype, [None]+self.data_dim, name='co_input')
			self.train_phase = tf.placeholder(tf.bool, name='phase')

			### build generator (encoder)
			self.g_layer = self.build_gen(self.im_input, self.train_phase)
			
			### build reconstructor (decoder)
			self.i_layer = self.build_rec(self.g_layer, self.train_phase)

			### build batch discriminator (critic)
			self.r_logits, self.r_hidden = self.build_dis(self.co_input, self.train_phase)
			self.g_logits, self.g_hidden = self.build_dis(self.g_layer, self.train_phase, reuse=True)

			### build batch attention, shape: (B, k, k, 1)
			self.r_att = self.build_att(self.r_hidden, self.train_phase)
			self.g_att = self.build_att(self.g_hidden, self.train_phase, reuse=True)
			print '>>> r_att shape: ', self.r_att.get_shape().as_list()
			print '>>> g_att shape: ', self.g_att.get_shape().as_list()

			### build real attention ground truth (center 1 hot)
			inds = tf.shape(self.r_att) / 2
			r_att_shape = tf.shape(self.r_att)
			updates = tf.constant([1.0])
			r_att_gt = tf.scatter_nd([inds[1:3]], updates, r_att_shape[1:3])
			r_att_gt = tf.reshape(r_att_gt, [1, r_att_shape[1], r_att_shape[2], 1])
			print '>>> r_att_gt shape: ', self.r_att_gt.get_shape().as_list()

			### real gen manifold interpolation
			int_rand = tf.random_uniform(tf.shape(g_layer)[0], dtype=tf_dtype)
			int_rand = tf.reshape(int_rand, [-1, 1, 1, 1])
			rg_layer = (1.0 - int_rand) * self.g_layer + int_rand * self.co_input
			self.rg_logits, _ = self.build_dis(rg_layer, self.train_phase, reuse=True)

			### build d losses
			if self.d_loss_type == 'log':
				self.d_r_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.r_logits, labels=tf.ones_like(self.r_logits, tf_dtype))
				self.d_g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
				self.d_rg_loss = tf.nn.sigmoid_cross_entropy_with_logits(
						logits=self.rg_logits, labels=tf.ones_like(self.rg_logits, tf_dtype))
			elif self.d_loss_type == 'was':
				self.d_r_loss = -self.r_logits 
				self.d_g_loss = self.g_logits
				self.d_rg_loss = -self.rg_logits
			else:
				raise ValueError('>>> d_loss_type: %s is not defined!' % self.d_loss_type)
			print '>>> d_r_loss shape: ', self.d_r_loss.get_shape().as_list()
			print '>>> d_g_loss shape: ', self.d_g_loss.get_shape().as_list()

			### gradient penalty
			### NaN free norm gradient
			rg_grad = tf.gradients(self.rg_logits, rg_layer)
			rg_grad_flat = tf.reshape(rg_grad, [-1, np.prod(self.data_dim)])
			rg_grad_ok = tf.reduce_sum(tf.square(rg_grad_flat), axis=1) > 1.
			rg_grad_safe = tf.where(rg_grad_ok, rg_grad_flat, tf.ones_like(rg_grad_flat))
			#rg_grad_abs = tf.where(rg_grad_flat >= 0., rg_grad_flat, -rg_grad_flat)
			rg_grad_abs =  0. * rg_grad_flat
			rg_grad_norm = tf.where(rg_grad_ok, 
				tf.norm(rg_grad_safe, axis=1), tf.reduce_sum(rg_grad_abs, axis=1))
			gp_loss = tf.square(rg_grad_norm - 1.0)
			### for logging
			self.rg_grad_norm_output = tf.norm(rg_grad_flat, axis=1)
			
			### d loss combination **weighted**
			self.d_loss_mean = tf.reduce_mean(
				tf.reduce_sum(self.r_att_gt * self.d_r_loss, axis=[1,2,3]) + \
				tf.reduce_sum(self.g_att * self.d_g_loss, axis=[1,2,3]))

			self.d_loss_total = self.d_loss_mean + \
				self.gp_loss_weight * tf.reduce_mean(gp_loss) ## enforcing gp everywhere

			### build g loss
			if self.g_loss_type == 'log':
				self.g_loss = -tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.zeros_like(self.g_logits, tf_dtype))
			elif self.g_loss_type == 'mod':
				self.g_loss = tf.nn.sigmoid_cross_entropy_with_logits(
					logits=self.g_logits, labels=tf.ones_like(self.g_logits, tf_dtype))
			elif self.g_loss_type == 'was':
				self.g_loss = -self.g_logits
			else:
				raise ValueError('>>> g_loss_type: %s is not defined!' % self.g_loss_type)

			### g loss mean **weighted**
			self.g_loss_mean = tf.reduce_mean(
				tf.reduce_sum(self.g_att * self.g_loss, axis=[1,2,3]), axis=None)

			### reconstruction loss mean **weighted** (upsampling with deconv: d_h=num_disc_convs*2 k_h=k_disc+(k-1)*num_disc_convs)
			#self.g_att_us = tf.image.resize_nearest_neighbor(self.g_att, tf.shape(self.i_layer)[1:3])
			k_init = tf.constant_initializer(1.0)
			self.g_att_us = conv2d_tr(self.g_att, 1, k_h=29, k_w=29, d_h=8, d_w=8, scope='rec_deconv', k_init=k_init, trainable=False)
			self.rec_loss_mean = tf.reduce_mean(tf.reduce_sum(
					self.g_att_us * tf.square(self.i_layer - self.co_input), axis=[1,2,3]))

			#self.g_grad_norm = tf.norm(tf.reshape(
			#	tf.gradients(self.g_loss, self.g_layer), [-1, np.prod(self.data_dim)]), axis=1)

			### g loss combination
			self.g_loss_total = self.g_loss_mean + self.rec_loss_weight * self.rec_loss_mean

			### collect params
			self.g_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "g_net")
			self.d_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "d_net")
			self.a_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "a_net")
			self.i_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, "i_net")

			### compute stat of weights
			self.nan_vars = 0.
			self.inf_vars = 0.
			self.zero_vars = 0.
			self.big_vars = 0.
			self.count_vars = 0
			for v in self.g_vars + self.d_vars + self.a_vars + self.i_vars:
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
			for v in self.g_vars:
				self.g_vars_count += int(np.prod(v.get_shape()))
			for v in self.d_vars:
				self.d_vars_count += int(np.prod(v.get_shape()))
			for v in self.a_vars:
				self.a_vars_count += int(np.prod(v.get_shape()))
			for v in self.i_vars:
				self.i_vars_count += int(np.prod(v.get_shape()))

			### build optimizers
			update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
			print '>>> update_ops list: ', update_ops
			with tf.control_dependencies(update_ops):
				self.g_opt = tf.train.AdamOptimizer(
					self.g_lr, beta1=self.g_beta1, beta2=self.g_beta2).minimize(
					self.g_loss_total, var_list=self.g_vars+self.a_vars+self.i_vars)
				self.d_opt = tf.train.AdamOptimizer(
					self.d_lr, beta1=self.d_beta1, beta2=self.d_beta2).minimize(
					self.d_loss_total, var_list=self.d_vars)
				#self.a_opt = tf.train.AdamOptimizer(
				#	self.a_lr, beta1=self.a_beta1, beta2=self.a_beta2).minimize(
				#	self.g_loss_total, var_list=self.a_vars)
				#self.i_opt = tf.train.AdamOptimizer(
				#	self.i_lr, beta1=self.i_beta1, beta2=self.i_beta2).minimize(
				#	self.rec_loss, var_list=self.i_vars)

			### summaries **g_num**
			g_loss_sum = tf.summary.scalar("g_loss", self.g_loss_mean)
			d_loss_sum = tf.summary.scalar("d_loss", self.d_loss_mean)
			rec_loss_sum = tf.summary.scalar("e_loss", self.rec_loss_mean)
			self.summary = tf.summary.merge([g_loss_sum, d_loss_sum, rec_loss_sum])

	def build_gen(self, z, train_phase):
		act = self.g_act
		with tf.variable_scope('g_net'):
			h1 = act(conv2d(z, 32, scope='conv1'))
			h2 = act(conv2d(h1, 32, scope='conv2'))
			h3 = conv2d(h2, 3, scope='conv3')
			o = tf.tanh(z + h3)
		return o

	def build_rec(self, x, train_phase):
		act = self.rec_act
		with tf.variable_scope('i_net'):
			h1 = act(conv2d(x, 32, scope='conv1'))
			h2 = act(conv2d(h1, 32, scope='conv2'))
			h3 = conv2d(h2, 3, scope='conv3')
			o = tf.tanh(z + h3)
		return o

	def build_dis(self, data_layer, train_phase, reuse=False):
		act = self.d_act
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('d_net'):
			### encoding the 64*64*3 image with conv into 8*8*1
			h1 = act(conv2d(data_layer, 32, d_h=2, d_w=2, scope='conv1', reuse=reuse))
			h2 = act(conv2d(h1, 64, d_h=2, d_w=2, scope='conv2', reuse=reuse))
			h3 = act(conv2d(h2, 128, d_h=2, d_w=2, scope='conv3', reuse=reuse))
			o = conv2d(h3, 1, k_h=1, k_w=1, scope='conv4', reuse=reuse)
		return o, h3

	def build_att(self, hidden_layer, act, train_phase, reuse=False):
		act = self.a_act
		bn = tf.contrib.layers.batch_norm
		with tf.variable_scope('a_net'):
			h1 = act(conv2d(hidden_layer, 128, k_h=1, k_w=1, scope='conv1', reuse=reuse))
			o = conv2d(h1, 1, k_h=1, k_w=1, scope='conv2', reuse=reuse)
			o_soft = tf.reshape(tf.nn.softmax(tf.contrib.layers.flatten(o)), tf.shape(o))
		return o_soft

	def start_session(self):
		self.saver = tf.train.Saver(tf.global_variables(), 
			keep_checkpoint_every_n_hours=1, max_to_keep=5)
		self.writer = tf.summary.FileWriter(self.log_dir, self.sess.graph)

	def save(self, fname):
		self.saver.save(self.sess, fname)

	def load(self, fname):
		self.saver.restore(self.sess, fname)

	def write_sum(self, sum_str, counter):
		self.writer.add_summary(sum_str, counter)

	def step(self, co_data, im_data=None, gen_update=False, 
		gen_only=False, stats_only=False, 
		att_only_co=False, att_only_im=False):
		batch_size = co_data.shape[0]		
		im_data = im_data.astype(np_dtype) if im_data is not None else None
		co_data = co_data.astype(np_dtype) if co_data is not None else None

		if stats_only:
			res_list = [self.nan_vars, self.inf_vars, self.zero_vars, self.big_vars]
			res_list = self.sess.run(res_list, feed_dict={})
			return res_list

		### only forward attention on co_data
		if att_only_co:
			feed_dict = {self.co_input: co_data, self.train_phase: False}
			res_list = self.sess.run(self.g_att, feed_dict=feed_dict)
			return res_list

		### only forward attention on im_data
		if att_only_im:
			feed_dict = {self.im_input: im_data, self.train_phase: False}
			res_list = self.sess.run(self.r_att, feed_dict=feed_dict)
			return res_list

		### only forward generator on z
		if gen_only:
			feed_dict = {self.co_input: co_data, self.train_phase: False}
			g_layer = self.sess.run(self.g_layer, feed_dict=feed_dict)
			return g_layer

		### run one training step on discriminator, otherwise on generator, and log **g_num**
		feed_dict = {self.co_input: co_data, self.im_input: im_data, self.train_phase: True}
		if not gen_update:
			res_list = [self.g_layer, self.summary, self.d_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		else:
			res_list = [self.g_layer, self.summary, self.g_opt]
			res_list = self.sess.run(res_list, feed_dict=feed_dict)
		
		### return summary and g_layer
		return res_list[1], res_list[0]