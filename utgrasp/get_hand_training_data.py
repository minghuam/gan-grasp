import sys,os,cv2,shutil

'''
index = 0
for d in sorted(os.listdir('hand')):
	for img in sorted(os.listdir(os.path.join('hand', d))):
		if index % 9 != 0:
			index += 1
			continue
		index += 1
		I = cv2.imread(os.path.join('hand', d, img))
		I = cv2.cvtColor(I, cv2.COLOR_BGR2GRAY)
		cv2.imshow('I', I)
		if cv2.waitKey(10) & 0xFF == 27:
			sys.exit(0)	
		cv2.imwrite(os.path.join('msk', d + '_' + img), I)	
'''

msks = sorted(os.listdir('msk'))
images = dict()
for d in os.listdir('images'):
	for img in os.listdir(os.path.join('images', d)):
		images[img] = os.path.join('images', d, img)
		print img

for msk in msks:
	print msk,'444'
	print images[msk]
	shutil.copy(images[msk], 'img')




