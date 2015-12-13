caffe_root = '/home/minghuam/caffe-fcn'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import cv2
import numpy as np
import argparse
import pickle

def ls_images(folder_path, extension = '.png'):
	return sorted([os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith(extension)])

def mkdir_safe(path):
	if not os.path.exists(path):
		os.mkdir(path)

target_index = int(sys.argv[1])

input_dir = '/home/klab/Documents/torch-gan-hand/gen_grasps/'


caffe.set_mode_gpu()
caffe.set_device(1)
net_proto_file =  'deploy.prototxt'
model_file = 'model/GRASP_iter_4000.caffemodel'

net = caffe.Net(net_proto_file, model_file, caffe.TEST)

Imean = np.load('image_mean.npy')
print Imean.shape
Imean = np.repeat(Imean.reshape((1,) + Imean.shape), repeats = 100, axis = 0)


input_images = ls_images(input_dir, '.png')
save_index = 0
Isave = np.zeros((640, 640, 3), np.uint8)
for img in input_images:
	print img
	Iraw = cv2.imread(img)
	I = cv2.resize(Iraw, (256*10, 256*10))
	I = I.reshape((10, 256, 2560, 3)).transpose((0,2,1,3))
	I = I.reshape((100, 256, 256, 3)).transpose((0,2,1,3))

	Icrop = I[:, 16:-16, 16:-16, :]
	Inorm = Icrop - Imean

	net.blobs['data'].data[...] = Inorm.transpose((0, 3, 1, 2))
	out = net.forward()
	e_score = np.exp(out['score'])
	e_score_sum = e_score.sum(axis = 1).reshape((-1,1))
	score = e_score/e_score_sum

	indices = np.argmax(score, axis = 1)
	for i in np.where(indices == target_index)[0].ravel():
		if score[i, target_index] < 0.12:
			continue
		print save_index, score[i, target_index]

		row = save_index/10
		col = save_index%10
		Isave[row*64:(row+1)*64, col*64:(col+1)*64, :] = cv2.resize(I[i,...], (64, 64))
		save_index += 1

		if save_index == 100:
			break

		cv2.imshow('I', Iraw)
		cv2.imshow('Isave', Isave)
		cv2.waitKey(20)

	if save_index == 100:
		break

cv2.imwrite('{}.png'.format(target_index), Isave)
