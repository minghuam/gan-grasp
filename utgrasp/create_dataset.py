import os,sys,cv2
import numpy as np
from skimage.transform import resize
import h5py


def to_bc01(b01c):
    return np.transpose(b01c, (0, 3, 1, 2))


def to_b01c(bc01):
    return np.transpose(bc01, (0, 2, 3, 1))

from PIL import Image
from skimage.transform import resize

if __name__ == '__main__':

    imgs_dir = '/home/minghuam/Desktop/ut_grasp/grasp_training/data/'
    imgs = []
    for d in os.listdir(imgs_dir):
        for img in os.listdir(os.path.join(imgs_dir, d)):
            I = np.array(Image.open(os.path.join(imgs_dir, d, img)))
            I = resize(I, (64, 64, 3), order = 3)
            print I.shape, I.dtype
            cv2.imshow('I',I)
            if cv2.waitKey(10) & 0xFF == 27:
                sys.exit(0)
            imgs.append(I)

    imgs = np.array(imgs)
    print '##################'
    print len(imgs)
    print imgs[0].shape

    idxs = np.random.permutation(np.arange(len(imgs)))
    imgs = imgs[idxs]
    imgs = to_bc01(imgs)
    print imgs.shape

    print 'saving hdf5'
    f = h5py.File('grasp.hdf5', 'w')
    f.create_dataset('grasp', data=imgs)
    f.close()