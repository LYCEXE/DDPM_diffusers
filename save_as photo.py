import numpy as np
# from scipy.misc import imsave
from imageio import  imsave

def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict

filename = 'dataset/cifar-10-batches-py'

meta = unpickle(filename+'/batches.meta')
label_name = meta[b'label_names']

for i in range(1,6):
	content = unpickle(filename+'/data_batch_'+str(i))
	print('load data...')
	print(content.keys())
	print('tranfering data_batch' + str(i))
	for j in range(10000):
		img = content[b'data'][j]
		img = img.reshape(3,32,32)
		img = img.transpose(1,2,0)
		# img_name = 'cifar10_fid/'+label_name[content[b'labels'][j]].decode() + '/batch_' + str(i) + '_num_' + str(j) +'.jpg'
		img_name = 'CIFAR10/'+ '/batch_' + str(i) + '_num_' + str(j) +'.jpg'
		imsave(img_name,img)