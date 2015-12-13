import sys,os,cv2,shutil

images_dir = 'images'
label_dir = 'label'
output_dir = 'grasp_images'

if not os.path.exists(output_dir):
	os.mkdir(output_dir)

grasp_frames = dict()
for label_file in os.listdir(label_dir):
	video = os.path.splitext(label_file)[0]
	with open(os.path.join(label_dir, label_file), 'r') as fr:
		for line in fr.readlines():
			tokens = line.strip().split('\t')
			grasp = tokens[0]
			start = int(tokens[1])
			end = int(tokens[2])
			if grasp not in grasp_frames:
				grasp_frames[grasp] = list()
			grasp_frames[grasp].append((video, start, end))
print grasp_frames

for grasp in grasp_frames:
	d = os.path.join(output_dir, grasp)
	print d
	if not os.path.exists(d):
		os.mkdir(d)
	for (video, start, end) in grasp_frames[grasp]:
		print video,start,end
		images = sorted([os.path.join(images_dir, video, f) for f in os.listdir(os.path.join(images_dir, video))])
		grasp_images = images[start-1:end]
		for img in grasp_images:
			shutil.copy(img, d)