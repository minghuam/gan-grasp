caffe_root = '/home/minghuam/caffe-dev/'
import sys,os
sys.path.insert(0, os.path.join(caffe_root, 'python'))
import caffe
import argparse
import numpy as np

base_weights = "VGG_CNN_M.caffemodel"
solver_prototxt = 'solver.prototxt'

# init
caffe.set_mode_gpu()
caffe.set_device(1)

solver = caffe.SGDSolver(solver_prototxt)
solver.net.copy_from(base_weights)

#solver.step(3000)
solver.step(4000)
