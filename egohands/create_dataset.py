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

    img_dir = 'hand_imgs'
    imgs = []

    Imean = cv2.imread('mean.jpg').astype(np.float64)

    for img in os.listdir(img_dir):
        '''2
        I = cv2.imread(os.path.join(img_dir, img))
        I = cv2.resize(I, (64, 64))
        #imgs.append(I.astype(np.float64) - Imean)
        #imgs.append(I.astype(np.float64))
        Ib = I[...,0]
        Ir = I[...,2]
        I[...,0] = Ir
        I[...,2] = Ib
        cv2.imshow('I', I)
        cv2.waitKey(10)
        '''
        I = np.array(Image.open(os.path.join(img_dir, img)))
        I = resize(I, (64, 64, 3), order = 3)
        print I.shape, I.dtype
        cv2.imshow('I',I)
        if cv2.waitKey(10) & 0xFF == 27:
            sys.exit(0)
        imgs.append(I)

    imgs = np.array(imgs)
    Isum = imgs.sum(axis = 0)/imgs.shape[0]
    print Isum.shape
    #cv2.imwrite('mean.jpg', np.clip(Isum, 0, 255).astype(np.uint8))

    print '##################'
    print len(imgs)
    print imgs[0].shape

    idxs = np.random.permutation(np.arange(len(imgs)))
    imgs = imgs[idxs]
    imgs = to_bc01(imgs)
    print imgs.shape

    print 'saving hdf5'
    f = h5py.File('hands3.hdf5', 'w')
    f.create_dataset('hands3', data=imgs)
    f.close()

    f = h5py.File('mean.hdf5', 'w')
    Imean = Imean.reshape((1,) + Imean.shape)
    Imean = to_bc01(Imean.astype(np.float32))
    print Imean.shape
    f.create_dataset('mean', data=Imean)
    f.close()