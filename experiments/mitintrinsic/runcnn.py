import sys
import os

import numpy as np
import scipy as sp

# Make sure that caffe is on the python path:
sys.path.append('python')
import caffe

def predict_thresholds(model_file, pretrained_weights, input_image):
    net = caffe.Classifier(model_file, pretrained_weights,
            mean=np.array([0.5, 0.5, 0.5]),
            channel_swap=(0, 1, 2),
            raw_scale=1,
            image_dims=(190, 190))

    net.set_phase_test()
    net.set_mode_cpu()

    predictions = net.dense_predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
    predictions = map(lambda p: p.squeeze(axis=(0, 1)), predictions)

    cnt = 0
    for p in predictions:
        caffe.io.save_image('testimg{0}.png'.format(cnt), p, scale=1.0)
        cnt = cnt + 1

    return predictions

