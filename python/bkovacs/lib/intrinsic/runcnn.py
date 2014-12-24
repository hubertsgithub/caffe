import sys
import os

import numpy as np

from lib.utils.misc.pathresolver import acrp

# Make sure that caffe is on the python path:
sys.path.append('python')
import caffe

def predict_thresholds(model_file, pretrained_weights, input_images):
    net = caffe.Classifier(model_file, pretrained_weights,
            mean=np.array([0.5]),
            raw_scale=1,
            image_dims=(190, 190))

    net.set_phase_test()
    net.set_mode_cpu()
    dirpath = acrp(os.path.join('experiments/cnn-input-output', net.name))
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

    for i, im in enumerate(input_images):
        caffe.io.save_image(os.path.join(dirpath, '{0}-input_image{1}.png'.format(predict_thresholds.counter, i)), im, scale=1.0)

    predictions = net.dense_predict(input_images)  # predict takes any number of images, and formats them for the Caffe net automatically
    predictions = map(lambda p: p.squeeze(axis=(0, 1)), predictions)

    cnt = 0
    for p in predictions:
        caffe.io.save_image(os.path.join(dirpath, '{0}-output_img{1}.png'.format(predict_thresholds.counter, cnt)), p, scale=1.0)
        cnt += 1

    predict_thresholds.counter += 1
    return predictions

predict_thresholds.counter = 0
