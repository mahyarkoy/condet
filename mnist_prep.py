import numpy as np
import gzip
import cPickle as pk
from skimage.transform import resize
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
mnist_path = '/media/evl/Public/Mahyar/Data/mnist.pkl.gz'

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
    
### read raw 28*28 mnist
f = gzip.open(mnist_path, 'rb')
train_set, val_set, test_set = pk.load(f)
f.close()

### reshape and resize mnist to 32*32
co_size = (32, 32, 3)
im_data = test_set[0]#train_set[0]
im_data = im_data.reshape((im_data.shape[0], 28, 28, 1))
im_data_re = np.zeros((im_data.shape[0], co_size[0], co_size[1], 1))
for i in range(im_data.shape[0]):
	im_data_re[i, ...] = im_sq_resize(im_data[i, ..., 0], (co_size[0], co_size[1], 1))
		#resize(im_data[i, ...], (co_size[0], co_size[1]), preserve_range=True)
#im_data_re = np.array(im_data_re)

### content data
co_data = np.tile(im_data_re, [1,1,1,3])

### generate random bg
sample_size = 100
im_size = (64, 64, 3)
im_bg = np.random.uniform(size=(sample_size,)+im_size)

### generate random bounding boxes with mnist digits
top_rand = np.random.randint(im_size[0]-co_size[0], size=sample_size)
left_rand = np.random.randint(im_size[1]-co_size[1], size=sample_size)
im_bboxes = list()
for i in range(sample_size):
    im_bg[i, top_rand[i]:top_rand[i]+co_size[0], left_rand[i]:left_rand[i]+co_size[1], ...] = \
        co_data[i%sample_size, ...]
    im_bboxes.append(
            np.array([left_rand[i], top_rand[i], 
                      left_rand[i]+co_size[1], top_rand[i]+co_size[0]]).reshape(1,4))
        
data_dir = '/media/evl/Public/Mahyar/Data'
### save to file
#with open(data_dir+'/mnist_test_ndarray.cpk', 'wb+') as fs:
#    pk.dump([im_bboxes, im_bg, co_data], fs)

fig, ax = plt.subplots(1)
ax.imshow(im_bg[0])
bbox = im_bboxes[0]
i = 0
rect = patches.Rectangle((bbox[i,0], bbox[i,1]), 
                         bbox[i,2]-bbox[i,0], 
                         bbox[i,3]-bbox[i,1], 
                         linewidth=1, edgecolor='r', facecolor='none')
ax.add_patch(rect)
plt.show()