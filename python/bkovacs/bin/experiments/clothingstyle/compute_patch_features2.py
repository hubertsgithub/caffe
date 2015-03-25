import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import pylab

from lib.utils.data import common
from lib.utils.misc import packer
from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.plothelper import plot_and_save_2D_array
from lib.utils.misc.progressbaraux import progress_bar
from lib.utils.net import visnet
from lib.utils.net.misc import init_net

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe


ROOTPATH = acrp('data/clothingstyle')
EXPROOTPATH = acrp('experiments/clothingstyle/')
TESTFILE = 'balanced/test.txt'

SAMPLESTART = 0
SAMPLECOUNT = 500000
STEPCOUNT = 500
REQUIREDPRECISION = 0.9


def compute_features(net, inputs, feature_options, croplen):
    input_dict = {}
    for input_name, input_image in inputs.iteritems():
        h, w = input_image.shape[0:2]
        cropw = w / 2
        croph = h / 2
        patch = common.crop_image(input_image, cropw, croph, croplen)

        caffe_in = net.preprocess(input_name, patch)
        input_dict[input_name] = caffe_in[np.newaxis, :, :, :]

    prediction = net.forward_all(blobs=feature_options, **input_dict)

    return prediction

def find_nearest_idx(arr, value):
    if len(arr) == 0:
        raise ValueError('Empty array provided!')

    if arr.ndim != 1:
        raise ValueError('Array should be 1D!')

    return np.argmin(np.abs(arr - value))

def compute_confusionmx(dists_equal, dists_notequal, threshold):
    # x: actual, y: predicted
    # 0: not equal, 1: equal
    confmx = np.zeros((2, 2))

    for d in dists_equal:
        if d < threshold:
            confmx[1, 1] += 1
        else:
            confmx[1, 0] += 1

    for d in dists_notequal:
        if d >= threshold:
            confmx[0, 0] += 1
        else:
            confmx[0, 1] += 1

    return confmx


def compute_accuracy(confmx):
    # (true positive + true negative) / total population
    return float(confmx[0, 0] + confmx[1, 1]) / np.sum(confmx)


def compute_recall(confmx):
    # true positive / condition positive
    sum = np.sum(confmx[1, :])
    if sum == 0.0:
        return 0.0
    else:
        return float(confmx[1, 1]) / sum


def compute_precision(confmx):
    # true positive / test outcome positive
    sum = np.sum(confmx[:, 1])
    if sum == 0.0:
        return 1.0
    else:
        return float(confmx[1, 1]) / sum


def process_line(line):
    tokens = l.split(' ')
    img_path1, img_path2, sim = tokens
    img_path1 = img_path1.strip()
    img_path2 = img_path2.strip()
    sim = sim.strip()

    sim = {'0': False, '1': True}.get(sim)
    if sim is None:
        raise ValueError('Invalid value for sim, it should be either \'0\' or \'1\'')

    return img_path1, img_path2, sim

if __name__ == '__main__':

    with open(os.path.join(ROOTPATH, TESTFILE), 'r') as f:
        lines = f.readlines()

    #create list of file names
    file_names_list = []
    for i,line in enumerate(lines):
        strs = line.split(None, 2)[:2]
        file_names_list.append(strs[0])
        file_names_list.append(strs[1])
    #filter for duplicates
    file_names_list = list(set(file_names_list))
    file_names_list = sorted(file_names_list)
    file_names_list = file_names_list[SAMPLESTART:SAMPLECOUNT]
    file_names = np.array(file_names_list)
    filenamesfilepath = os.path.join(EXPROOTPATH, 'image_filenames{0}-{1}.npy'.format(SAMPLESTART, SAMPLESTART+SAMPLECOUNT))
    if os.path.exists(filenamesfilepath):
        pass
    else:
        np.save(filenamesfilepath, file_names)

    network_options = {}

    input_names = ['data']
    # raw_scale is used to upscale the loaded image from [0, 1] (because we
    # load it that way) to [0, 255]
    # input_scale is used to downscale the image after mean subraction, but
    # because there networks were trained with [0, 255] output, we set this to
    # 1.
    input_config = {input_name: {'channel_swap': (2, 1, 0), 'raw_scale': 255., 'input_scale': 1.} for input_name in input_names}
    mean = [104.0, 117.0, 123.0]

    # model_file = acrp('models/bvlc_alexnet/deploy.prototxt')
    # pretrained_weights = acrp('models/bvlc_alexnet/bvlc_alexnet.caffemodel')
    # print 'Initializing alexnet net'
    # net = init_net(model_file, pretrained_weights, mean, input_config)
    # network_options['alexnet'] = {'feature_options': ['fc7'], 'net': net, 'croplen': 227, 'input_names': input_names, 'comp_feature_func': compute_features}

    # model_file = acrp('ownmodels/clothingstyle/deploy_alexnet-siamese.prototxt')
    # pretrained_weights = acrp('ownmodels/clothingstyle/snapshots/caffenet_train_alexnet-siamese-base_lr5e-06_iter_90000.caffemodel')
    # print 'Initializing alexnet-siamese net'
    # net = init_net(model_file, pretrained_weights, mean, input_config)
    # network_options['alexnet-siamese'] = {'feature_options': ['embedding'], 'net': net, 'croplen': 227, 'input_names': input_names, 'comp_feature_func': compute_features}

    model_file = acrp('ownmodels/clothingstyle/deploy_googlenet-siamese.prototxt')
    pretrained_weights = acrp('ownmodels/clothingstyle/snapshots/caffenet_train_googlenet-siamese-base_lr1e-05_iter_130000.caffemodel')
    print 'Initializing googlenet-siamese net'
    net = init_net(model_file, pretrained_weights, mean, input_config)
    network_options['googlenet-siamese'] = {'feature_options': ['embedding'], 'net': net, 'croplen': 224, 'input_names': input_names, 'comp_feature_func': compute_features}

    features = []

    for net_name, net_options in network_options.iteritems():
        feature_options = net_options['feature_options']
        net = net_options['net']
        croplen = net_options['croplen']
        input_names = net_options['input_names']
        comp_feature_func = net_options['comp_feature_func']

        #last_grayimg_path = ''
        for i, img_path in enumerate(progress_bar(file_names_list)):
            if i % 20000 == 0 and i>0:
                image_features = np.array(features)            
                featurefilepath = os.path.join(EXPROOTPATH, 'image_features{0}-{1}.npy'.format(SAMPLESTART, SAMPLESTART+i-1))
                if os.path.exists(featurefilepath):
                    pass
                else:
                    np.save(featurefilepath, image_features[0:image_features.shape[0]])

            img  = common.load_image(img_path, is_srgb=False)

            # We list all possible inputs here...
            inputs = {}
            inputs['data'] = img
            outputblobs = comp_feature_func(net, inputs, feature_options, croplen)

            for f in feature_options:
                f1 = outputblobs[f]
                f1 = np.ravel(np.squeeze(f1))

                features.append(f1)
    image_features = np.array(features)            
    featurefilepath = os.path.join(EXPROOTPATH, 'image_features{0}-{1}.npy'.format(SAMPLESTART, SAMPLESTART+SAMPLECOUNT-1))
    if os.path.exists(featurefilepath):
        pass
    else:
        np.save(featurefilepath, image_features)

