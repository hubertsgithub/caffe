import sys
import os

import numpy as np
import matplotlib.pyplot as plt
import pylab
import random

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.plothelper import plot_and_save_2D_array
from lib.utils.misc.progressbaraux import progress_bar
from lib.utils.data import common
from lib.utils.misc import packer
from lib.utils.net.misc import init_net
from lib.utils.net import visnet

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe


#CROPLEN = 224
#MODEL_FILE = acrp('models/vgg_cnn_m/VGG_CNN_M_deploy.prototxt')
#PRETRAINED_WEIGHTS = acrp('models/vgg_cnn_m/VGG_CNN_M.caffemodel')
#MEAN_FILE = acrp('models/vgg_cnn_m/VGG_mean.binaryproto')

#CROPLEN = 32
#MODEL_FILE = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small.prototxt')
#PRETRAINED_WEIGHTS = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese2-base_lr0.005_iter_20000.caffemodel')
#MEAN_FILE = 128

#CROPLEN = 32
#MODEL_FILE = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small_finetune.prototxt')
#PRETRAINED_WEIGHTS = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese_finetune-base_lr0.0001_iter_2500.caffemodel')
#MEAN_FILE = 128

ROOTPATH = acrp('data/iiw-dataset')
EXPROOTPATH = acrp('experiments/distancemetrics')

SAMPLECOUNT = 1000
STEPCOUNT = 500
REQUIREDPRECISION = 0.9


def test_alexnet():
    model_file = acrp('models/bvlc_reference_caffenet/deploy.prototxt')
    pretrained_weights = acrp('models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel')

    visnet.vis_net(acrp('models/alma'), '', model_file, pretrained_weights)

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


def compute_features(net, inputs, feature_options, px, py, croplen):
    input_dict = {}
    for input_name, input_image in inputs.iteritems():
        h, w = input_image.shape[0:2]
        cropw = px * w
        croph = py * h
        patch = common.crop_image(input_image, cropw, croph, croplen)

        caffe_in = net.preprocess(input_name, patch)
        input_dict[input_name] = caffe_in[np.newaxis, :, :, :]

    prediction = net.forward_all(blobs=feature_options, **input_dict)

    return prediction


def compute_chrompatch(net, inputs, feature_options, px, py, croplen):
    h, w = inputs['chrom'].shape[0:2]
    centerw = px * w
    centerh = py * h

    ret = {}
    for f in feature_options:
        shift = (f - 1) / 2
        fet = np.zeros((f, f, 3))
        fet[:] = inputs['chrom'][centerh-shift:centerh+shift+1, centerw-shift:centerw+shift+1]

        ret[f] = fet

    return ret


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

    hist_min = min([np.min(dists_equal), np.min(dists_notequal)])
    hist_max = max([np.max(dists_equal), np.max(dists_notequal)])
    heq, _ = np.histogram(dists_equal, bins=50, range=(hist_min, hist_max))
    hneq, _ = np.histogram(dists_notequal, bins=50, range=(hist_min, hist_max))
    y_max = max([np.max(heq), np.max(hneq)])

    plt.subplot(2, 1, 1)
    plt.hist(dists_equal, bins=50, range=(hist_min, hist_max), color='r')
    plt.ylim((0, y_max))
    plt.title(distmetricname)
    plt.ylabel('Count')
    plt.legend(['Equal distances'], loc='best')

    plt.subplot(2, 1, 2)
    plt.hist(dists_notequal, bins=50, range=(hist_min, hist_max), color='b')
    plt.ylim((0, y_max))
    plt.xlabel('Distance')
    plt.ylabel('Count')
    plt.legend(['Not equal distances'], loc='best')

    plt.savefig(os.path.join(EXPROOTPATH, '{0}-dist-histograms.png'.format(distmetricname)))
    plt.clf()

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

    return thresacc, acc, thresprec, prec, recall


if __name__ == '__main__':
    #test_alexnet()

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

    network_options = {}
    #network_options['vgg'] = {'feature_options': ['fc7', 'pool5'], 'comp_feature_func': compute_features}

    input_names = ['data', 'chrom']
    input_config = {input_name: {'channel_swap': (2, 1, 0), 'raw_scale': 255., 'input_scale': 1./255.} for input_name in input_names}
    croplen = 32
    model_file = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small.prototxt')
    pretrained_weights = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese2-base_lr0.005_iter_20000.caffemodel')
    mean = 128
    print 'Initializing siamese net'
    net = init_net(model_file, pretrained_weights, mean, input_config)
    network_options['siamese'] = {'feature_options': ['fc2'], 'net': net, 'croplen': croplen, 'input_names': input_names, 'comp_feature_func': compute_features}

    input_names = ['data', 'chrom']
    input_config = {input_name: {'channel_swap': (2, 1, 0), 'raw_scale': 255., 'input_scale': 1./255.} for input_name in input_names}
    croplen = 32
    model_file = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small_chrom.prototxt')
    pretrained_weights = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese_chrom-base_lr0.0001_iter_15000.caffemodel')
    mean = 128
    print 'Initializing siamese_chrom net'
    net = init_net(model_file, pretrained_weights, mean, input_config)
    network_options['siamese_chrom'] = {'feature_options': ['fc2'], 'net': net, 'croplen': croplen, 'input_names': input_names, 'comp_feature_func': compute_features}

    input_names = ['data', 'chrom']
    input_config = {input_name: {'channel_swap': (2, 1, 0), 'raw_scale': 255., 'input_scale': 1./255.} for input_name in input_names}
    croplen = 32
    model_file = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small_chrom.prototxt')
    pretrained_weights = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese_chrom_margin5000-base_lr0.0001_iter_20000.caffemodel')
    mean = 128
    print 'Initializing siamese_chrom_margin5000 net'
    net = init_net(model_file, pretrained_weights, mean, input_config)
    network_options['siamese_chrom_margin5000'] = {'feature_options': ['fc2'], 'net': net, 'croplen': croplen, 'input_names': input_names, 'comp_feature_func': compute_features}

    input_names = ['data', 'chrom']
    input_config = {input_name: {'channel_swap': (2, 1, 0), 'raw_scale': 255., 'input_scale': 1./255.} for input_name in input_names}
    croplen = 32
    model_file = acrp('ownmodels/nonlocalreflnet/deploy_siamese_small_finetune_chrom.prototxt')
    pretrained_weights = acrp('ownmodels/nonlocalreflnet/snapshots/caffenet_train_nonlocalrefl_siamese_finetune_chrom-base_lr0.0001_iter_20000.caffemodel')
    mean = 128
    print 'Initializing siamese_finetune_chrom_margin5000 net'
    net = init_net(model_file, pretrained_weights, mean, input_config)
    network_options['siamese_finetune_chrom_margin5000'] = {'feature_options': ['fc2'], 'net': net, 'croplen': croplen, 'input_names': input_names, 'comp_feature_func': compute_features}

    network_options['zhao'] = {'feature_options': [3, 5], 'net': None, 'croplen': None, 'input_names': None, 'comp_feature_func': compute_chrompatch}

    scores = []
    indices = []
    features = {}
    distances_equal = {}
    distances_notequal = {}

    # Sort sampled lines, so we don't read the same image twice
    sampled_lines = sorted(sampled_lines, key=lambda l: l.split(' ')[0])

    for net_name, net_options in network_options.iteritems():
        feature_options = net_options['feature_options']
        net = net_options['net']
        croplen = net_options['croplen']
        input_names = net_options['input_names']
        comp_feature_func = net_options['comp_feature_func']

        features[net_name] = {}
        distances_equal[net_name] = {}
        distances_notequal[net_name] = {}
        for f in feature_options:
            features[net_name][f] = {}
            distances_equal[net_name][f] = []
            distances_notequal[net_name][f] = []

            for dm_idx, dm in enumerate(distmetrics):
                distances_equal[net_name][f].append([])
                distances_notequal[net_name][f].append([])

        last_grayimg_path = ''
        for l_idx, l in enumerate(progress_bar(sampled_lines)):
            #  f.write('{0} {1} {2} {3} {4} {5} {6}\n'.format(grayimg_path, chromimg_path, 1, p1x, p1y, p2x, p2y))
            tokens = l.split(' ')
            grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens
            sim = {'0': False, '1': True}.get(sim)
            p1x = float(p1x)
            p1y = float(p1y)
            p2x = float(p2x)
            p2y = float(p2y)

            # Don't load the image again if it is not necessary
            if grayimg_path != last_grayimg_path:
                head, filenameext = os.path.split(acrp(grayimg_path))
                origfilenameext = filenameext.replace('-gray', '-resized')
                origfilepath = os.path.join(head, origfilenameext)

                if not os.path.exists(origfilepath):
                    raise ValueError('Image file doesn\'t exist: {0}!'.format(origfilepath))

                img = common.load_image(origfilepath, is_srgb=True)
                chrom_img = common.compute_chromaticity_image(img)
                last_grayimg_path = grayimg_path

            # We list all possible inputs here...
            inputs = {}
            inputs['data'] = img
            inputs['chrom'] = chrom_img

            outputblobs1 = comp_feature_func(net, inputs, feature_options, p1x, p1y, croplen)
            outputblobs2 = comp_feature_func(net, inputs, feature_options, p2x, p2y, croplen)

            for f in feature_options:
                f1 = outputblobs1[f]
                f2 = outputblobs2[f]
                f1 = np.ravel(np.squeeze(f1))
                f2 = np.ravel(np.squeeze(f2))

                features[net_name][f][l_idx] = [f1, f2]

                for dm_idx, dm in enumerate(distmetrics):
                    dist = dm['func'](f1, f2)
                    if sim:
                        distances_equal[net_name][f][dm_idx].append(dist)
                    else:
                        distances_notequal[net_name][f][dm_idx].append(dist)

    packer.fpackb(features, 1.0, os.path.join(EXPROOTPATH, 'featuredata.dat'))

    for net_name, net_options in network_options.iteritems():
        print '############# {0} #############'.format(net_name)

        feature_options = net_options['feature_options']
        for f in feature_options:
            for dm_idx, dm in enumerate(distmetrics):
                indices.append((net_name, f, dm['name']))
                scores.append(analyze_distance_metric('{0}-{1}-{2}'.format(net_name, f, dm['name']), distances_equal[net_name][f][dm_idx], distances_notequal[net_name][f][dm_idx], STEPCOUNT, REQUIREDPRECISION))

    # Search for best results
    print '########### BEST RESULTS ###########'
    scores = np.array(scores)
    max_inds = np.argmax(scores, axis=0)
    net_name, f, dmname = indices[max_inds[1]]
    accuracy = scores[max_inds[1], 1]
    print 'All values: {0}'.format(scores[:, 1])
    print 'Best accuracy {0}: {1} network, {2} feature with {3} distance metric'.format(accuracy, net_name, f, dmname)
    net_name, f, dmname = indices[max_inds[4]]
    recall = scores[max_inds[4], 4]
    print 'All values: {0}'.format(scores[:, 4])
    print 'Best recall {0}: {1} network, {2} feature with {3} distance metric'.format(recall, net_name, f, dmname)



