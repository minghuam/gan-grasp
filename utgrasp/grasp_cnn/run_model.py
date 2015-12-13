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

parser = argparse.ArgumentParser()
parser.add_argument('test_file', help = 'Test file')
parser.add_argument('deploy_file', help = 'Deploy file')
parser.add_argument('model_file', help = 'Model file')
parser.add_argument('output_dir', help = 'Output dir')
parser.add_argument('n_iterations', help = 'iteration')
parser.add_argument('gpu_id', help = 'GPU id')
args = parser.parse_args()

caffe.set_mode_gpu()
caffe.set_device(int(args.gpu_id))
net_proto_file =  args.deploy_file

net = caffe.Net(net_proto_file, args.model_file, caffe.TEST)

crop_size = 224
#Imean = np.load('image_mean.npy').transpose(2, 1, 0)
Imean = np.load('image_mean.npy')
x = (Imean.shape[1] - crop_size)/2
y = (Imean.shape[0] - crop_size)/2
Imean = Imean[y:y+crop_size, x:x+crop_size, :]

video_images = dict()
with open(args.test_file, 'r') as fr:
	for line in fr.readlines():
		tokens = line.strip().split(' ')
		img = tokens[0]
		label = int(tokens[1])
		video = os.path.basename(os.path.dirname(img))
		if video not in video_images:
			video_images[video] = dict()
			video_images[video]['images'] = list()
			video_images[video]['label'] = label
		video_images[video]['images'].append(img)

predict_results = dict()
scores = dict()
n_videos = len(video_images)
for index, video in enumerate(video_images):
        print index+1, n_videos
	label = video_images[video]['label']
	predict_labels = dict()
	score = None
	for img in video_images[video]['images']:
		I = cv2.imread(img)
		x = (I.shape[1] - crop_size)/2
		y = (I.shape[0] - crop_size)/2
		I = I[y:y+crop_size, x:x+crop_size, :]
		Inorm = (I - Imean).reshape((1, I.shape[0], I.shape[1], I.shape[2]))
		net.blobs['data'].data[...] = Inorm.transpose(0, 3, 1, 2)
		out = net.forward()
		if score is None:
			score = out['score'].ravel()
			#score = net.blobs['fc7_gtea'].data.ravel()
		else:
			score += out['score'].ravel()
			#score += net.blobs['fc7_gtea'].data.ravel()
	score = score/len(video_images[video]['images'])
	scores[video] = (score, label)
	predict_label = np.argmax(score)
	print video, label, predict_label
	predict_results[video] = (predict_label, label)

save_file = os.path.splitext(os.path.basename(args.test_file))[0] + '_{}_results.pkl'.format(args.n_iterations)
pickle.dump(scores, open(os.path.join(args.output_dir, save_file), 'w'))

# overall accuracy
correct_count = 0
for video in predict_results:
	if predict_results[video][0] == predict_results[video][1]:
		correct_count += 1
print 'overall accuracy:',float(correct_count)/len(predict_results)

per_action_results = dict()
for video in predict_results:
	action = '_'.join(video.split('_')[1:-1])
	if action not in per_action_results:
		per_action_results[action] = list()
	per_action_results[action].append(predict_results[video])

per_action_accuracy = dict()
for action in per_action_results:
	total_count = len(per_action_results[action])
	correct_count = 0
	for (label, predict_label) in per_action_results[action]:
		if label == predict_label:
			correct_count += 1
	per_action_accuracy[action] = float(correct_count)/total_count

accuracies = per_action_accuracy.values()
print 'average accuracy:', sum(accuracies)/len(accuracies)
