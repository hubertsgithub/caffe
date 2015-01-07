import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar
from lib.utils.data import common
from lib.utils.misc import packer

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe

CROPLEN = 224
MODEL_FILE = acrp('models/vgg_cnn_m/VGG_CNN_M_deploy.prototxt')
PRETRAINED_WEIGHTS = acrp('models/vgg_cnn_m/VGG_CNN_M.caffemodel')
ROOTPATH = acrp('data/iiw-dataset')
MEAN_FILE = acrp('models/vgg_cnn_m/VGG_mean.binaryproto')
EXPROOTPATH = acrp('experiments/distancemetrics')

SAMPLECOUNT = 1000
STEPCOUNT = 500
REQUIREDPRECISION = 0.9


def test_alexnet():
    model_file = acrp('models/bvlc_reference_caffenet/deploy.prototxt')
    pretrained_weights = acrp('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')
    net = caffe.Net(model_file, pretrained_weights)
    net.set_phase_test()
    net.set_mode_cpu()
    input_name = 'data'
    net.set_channel_swap(input_name, (2, 1, 0))
    net.set_raw_scale(input_name, 255)

    meanarr = np.load(acrp('python/caffe/imagenet/ilsvrc_2012_mean.npy'))
    meanarr = np.squeeze(meanarr)

    net.set_mean(input_name, meanarr)

    # test for a cat image
    im = common.load_image(acrp('examples/images/cat.jpg'), is_srgb=False)
    im = common.resize_and_crop_image(im, 256, 227, keep_aspect_ratio=False, use_greater_side=False)
    common.print_array_info(im)
    caffe_in = net.preprocess(input_name, im)
    prediction = net.forward_all(blobs=None, **{input_name: caffe_in[np.newaxis, :, :, :]})['prob']
    prediction = np.squeeze(prediction)
    print 'prediction shape:', prediction.shape
    plt.plot(prediction)
    pylab.show()
    print 'predicted class:', prediction.argmax()


def compute_feature(net, input_name, input_image, px, py, croplen):
    h, w = input_image.shape[0:2]
    cropw = px * w
    croph = py * h
    patch = common.crop_image(input_image, cropw, croph, croplen)

    caffe_in = net.preprocess(input_name, patch)
    prediction = net.forward_all(blobs=['fc7'], **{input_name: caffe_in[np.newaxis, :, :, :]})

    return prediction


def init_net(model_file, pretrained_weights, mean_file, input_name):
    net = caffe.Net(model_file, pretrained_weights)
    net.set_phase_test()
    net.set_mode_cpu()
    net.set_channel_swap(input_name, (2, 1, 0))
    net.set_raw_scale(input_name, 255)

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    meanarr = caffe.io.blobproto_to_array(blob)
    # Remove the first dimension (batch), which is 1 anyway
    meanarr = np.squeeze(meanarr, axis=0)

    net.set_mean(input_name, meanarr)

    return net


def plot_and_save_2D_array(filename, arr, xlabel='', xinterval=None, ylabel='', yinterval=None):
    if arr.ndim != 2 or arr.shape[1] != 2:
        raise ValueError('The array should be 2D and the second dimension should be 2!')

    plt.plot(arr[:, 0], arr[:, 1])
    name, ext = os.path.splitext(os.path.basename(filename))
    plt.title(name)
    plt.xlabel(xlabel)
    if xinterval:
        plt.xlim(xinterval)

    if yinterval:
        plt.ylim(yinterval)

    plt.ylabel(ylabel)
    plt.savefig(filename)
    plt.clf()


def find_nearest_idx(arr, value):
    if len(arr) == 0:
        raise ValueError('Empty array provided!')

    if arr.ndim != 1:
        raise ValueError('Array should be 1D!')

    return np.argmin(np.abs(arr - value))


def solve_accuracy(distmetricname, dists_equal, dists_notequal, avg_equal, avg_notequal, stepcount, req_prec):
    accuracies = []
    recalls = []
    precisions = []
    for threshold in np.linspace(0, avg_notequal * 3, stepcount):
        confmx = compute_confusionmx(dists_equal, dists_notequal, threshold)
        acc = compute_accuracy(confmx)
        recall = compute_recall(confmx)
        prec = compute_precision(confmx)

        accuracies.append([threshold, acc])
        recalls.append([threshold, recall])
        precisions.append([threshold, prec])

    accuracies = np.array(accuracies)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    precrecall = np.empty_like(recalls)
    precrecall[:, 0] = recalls[:, 1]
    precrecall[:, 1] = precisions[:, 1]

    plot_and_save_2D_array(os.path.join(EXPROOTPATH, '{0}-threshold-accuracy.png'.format(distmetricname)), accuracies, xlabel='threshold', ylabel='accuracy', yinterval=(0.0, 1.0))
    plot_and_save_2D_array(os.path.join(EXPROOTPATH, '{0}-threshold-precision.png'.format(distmetricname)), precisions, xlabel='threshold', ylabel='precision', yinterval=(0.0, 1.0))
    plot_and_save_2D_array(os.path.join(EXPROOTPATH, '{0}-threshold-recall.png'.format(distmetricname)), recalls, xlabel='threshold', ylabel='recall', yinterval=(0.0, 1.0))
    plot_and_save_2D_array(os.path.join(EXPROOTPATH, '{0}-precision-recall.png'.format(distmetricname)), precrecall, xlabel='recall', xinterval=(0.0, 1.0), ylabel='precision', yinterval=(0.0, 1.0))

    best_index = np.argmax(accuracies[:, 1])
    idx = find_nearest_idx(precisions[:, 1], req_prec)
    print 'best_index: {0}, idx: {1}'.format(best_index, idx)
    # thresacc, acc, thresprec, prec, recall
    return accuracies[best_index, 0], accuracies[best_index, 1], precisions[idx, 0], precrecall[idx, 1], precrecall[idx, 0]


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


def analyze_distance_metric(distmetricname, dists_equal, dists_notequal, stepcount, req_prec):
    print '** {0} distance metric **'.format(distmetricname)
    avg_equal = np.mean(dists_equal)
    avg_notequal = np.mean(dists_notequal)
    print 'Average distance between patches with equal reflectance: {0}'.format(avg_equal)
    print 'Standard deviation: {0}'.format(np.std(dists_equal))
    print 'Average distance between patches with not equal reflectance: {0}'.format(avg_notequal)
    print 'Standard deviation: {0}'.format(np.std(dists_notequal))
    # Solve for accuracy
    thresacc, acc, thresprec, prec, recall = solve_accuracy(distmetricname, dists_equal, dists_notequal, avg_equal, avg_notequal, stepcount, req_prec)
    print 'Best accuracy: {0} at threshold: {1}'.format(acc, thresacc)
    print 'At ~ {0} precision ({1}) the recall is: {2} at threshold: {3}'.format(req_prec, prec, recall, thresprec)


if __name__ == '__main__':
    #test_alexnet()
    input_name = 'data'
    net = init_net(MODEL_FILE, PRETRAINED_WEIGHTS, MEAN_FILE, input_name)

    with open(os.path.join(ROOTPATH, 'train.txt'), 'r') as f:
        lines = f.readlines()

    # separate lines into equal/notequal
    eq_lines = []
    noteq_lines = []
    for l in lines:
        tokens = l.split(' ')
        grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens
        sim = {'0': False, '1': True}.get(sim)

        if sim:
            eq_lines.append(l)
        else:
            noteq_lines.append(l)

    eq_lines = random.sample(eq_lines, SAMPLECOUNT / 2)
    noteq_lines = random.sample(noteq_lines, SAMPLECOUNT / 2)
    sampled_lines = eq_lines + noteq_lines

    distmetrics = []
    distmetrics.append({'name': 'Euclidean', 'func': lambda f1, f2: np.linalg.norm(f1 - f2)})
    distmetrics.append({'name': 'Dot', 'func': lambda f1, f2: 1 - np.dot(f1, f2) / (np.linalg.norm(f1) * np.linalg.norm(f2))})
    distmetrics.append({'name': 'Chai', 'func': lambda f1, f2: np.sum(np.square(f1 - f2) / np.clip(f1 + f2, 1e-10, np.inf))})
    distmetrics.append({'name': 'L1', 'func': lambda f1, f2: np.sum(np.abs(f1 - f2))})
    distances_equal = []
    distances_notequal = []
    for dm_idx, dm in enumerate(distmetrics):
        distances_equal.append([])
        distances_notequal.append([])

    features = {}

    for l_idx, l in enumerate(progress_bar(sampled_lines)):
        #  f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(grayimg_path, chromimg_path, 1, p1x, p1y, p2x, p2y))
        tokens = l.split(' ')
        grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens
        sim = {'0': False, '1': True}.get(sim)
        p1x = float(p1x)
        p1y = float(p1y)
        p2x = float(p2x)
        p2y = float(p2y)

        head, filenameext = os.path.split(acrp(grayimg_path))
        origfilenameext = filenameext.replace('-gray', '-resized')
        origfilepath = os.path.join(head, origfilenameext)

        if not os.path.exists(origfilepath):
            raise ValueError('Image file doesn\'t exist: {0}!'.format(origfilepath))

        im = common.load_image(origfilepath, is_srgb=True)
        outputblobs1 = compute_feature(net, input_name, im, p1x, p1y, CROPLEN)
        outputblobs2 = compute_feature(net, input_name, im, p2x, p2y, CROPLEN)

        f1 = outputblobs1['fc7']
        f2 = outputblobs2['fc7']
        f1 = np.squeeze(f1)
        f2 = np.squeeze(f2)

        features[l] = [f1, f2]

        for dm_idx, dm in enumerate(distmetrics):
            dist = dm['func'](f1, f2)
            if sim:
                distances_equal[dm_idx].append(dist)
            else:
                distances_notequal[dm_idx].append(dist)

    packer.fpackb(features, 1.0, os.path.join(EXPROOTPATH, 'featuredata.dat'))

    for dm_idx, dm in enumerate(distmetrics):
        analyze_distance_metric(dm['name'], distances_equal[dm_idx], distances_notequal[dm_idx], STEPCOUNT, REQUIREDPRECISION)

