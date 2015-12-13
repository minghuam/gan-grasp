import os,sys,cv2,random

images_dir = '../hand_images'
output_dir = 'data'

images = dict()
for d in sorted(os.listdir(images_dir)):
	grasp = d.replace(' ', '_')
	if d not in images_dir:
		images[grasp] = list()
	for img in os.listdir(os.path.join(images_dir, d)):
		images[grasp].append(os.path.join(images_dir, d, img))

grasp_ids = dict()
index = 0
for g in images:
	grasp_ids[g] = index
	index += 1

with open('grasp_id.txt', 'w') as fw:
	for g in grasp_ids:
		fw.write(g + ' ' + str(grasp_ids[g]) + '\n')

fw_train = open('train_data.txt', 'w')
fw_test = open('test_data.txt', 'w')

for d in images:
	print d
	output_d = os.path.join(output_dir, d)
	if not os.path.exists(output_d):
		os.mkdir(output_d)
	imgs = list()
	for img in images[d]:
		I =  cv2.imread(img)
		I = cv2.resize(I, (64, 64))
		I = cv2.resize(I, (256, 256))
		out_img = os.path.join(output_d, os.path.basename(img))
		cv2.imwrite(out_img, I)
		imgs.append(out_img)
	random.shuffle(imgs)
	n = int(len(imgs) * 0.8)
	for i in range(n):
		fw_train.write(os.path.abspath(imgs[i]) + ' ' + str(grasp_ids[d]) + '\n')
	for i in range(n, len(imgs)):
		fw_test.write(os.path.abspath(imgs[i]) + ' ' + str(grasp_ids[d]) + '\n')

fw_train.close()
fw_test.close()