import sys,os,cv2
import scipy.io as sio
import numpy as np

images_dir = '_LABELLED_SAMPLES'
image_folders = sorted([os.path.join(images_dir, d) for d in os.listdir(images_dir)])
output_dir = 'hand_imgs'
if not os.path.exists(output_dir):
	os.mkdir(output_dir)

hand_index = 0
for folder in image_folders:
	files = [os.path.join(folder, f) for f in os.listdir(folder)]
	images = sorted([f for f in files if f.endswith('.jpg')])
	mat_file = [f for f in files if f.endswith('.mat')][0]


	m = sio.loadmat(mat_file)['polygons']
	print m.shape, len(images)

	index = 0
	for img in images:
		I = cv2.imread(img)

		# Igray = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		# ret,thresh = cv2.threshold(Igray,127,255,0)
		# cv2.imshow('t', thresh)
		# contours, hierarchy = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

		#for i in range(len(m[0, index])):
		for i in range(2, 4):
			if len(m[0, index][i]) > 1:
				contour = np.array(m[0, index][i]).reshape((-1,1,2)).astype(np.int32)
				
				# bounding box
				min_xy = contour.min(axis = 0)
				max_xy = contour.max(axis = 0)

				x1 = min_xy[0][0]
				y1 = min_xy[0][1]
				x2 = max_xy[0][0]
				y2 = max_xy[0][1]

				# center
				cx = (x1 + x2)/2
				cy = (y1 + y2)/2
				size = max(x2 - x1, y2 - y1)

				# crop region
				x1 = cx - size/2
				y1 = cy - size/2
				x2 = cx + size/2
				y2 = cy + size/2

				if x1 < 0 or y1 < 0:
					continue
				if x2 > I.shape[1] - 1 or y2 > I.shape[0] - 1:
					continue

				# area
				min_area = 6500
				max_area = 50000
				area = cv2.contourArea(contour)
				if area < min_area or area > max_area:
					continue
				print x1,x2,y1,y2
				Ihand = cv2.resize(I[y1:y2, x1:x2, :], (256, 256))
				#cv2.drawContours(I, [contour], 0, (0,255,0), -1)
				#cv2.rectangle(I, (x1,y1), (x2,y2), (0,0,255))
				#cv2.imshow('I', I)
				cv2.imshow('hand', Ihand)
				if cv2.waitKey(10) & 0xFF == 27:
					sys.exit(0)

				cv2.imwrite(os.path.join(output_dir, '{:06d}.jpg'.format(hand_index)), Ihand)
				hand_index += 1

		index += 1

		# cv2.imshow('I', I)
		# if cv2.waitKey(0) & 0xFF == 27:
		# 	sys.exit(0)

