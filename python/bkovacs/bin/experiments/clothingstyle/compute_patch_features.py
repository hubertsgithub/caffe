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
TESTFILE = 'balanced/train.txt'

SAMPLECOUNT = 100000
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


def analyze_distance_metric(distmetricname, dists_equal, dists_notequal, stepcount, req_prec, fout):
    fout.write('** {0} distance metric **\n'.format(distmetricname))
    #fout.write(str(dists_equal))
    #fout.write(str(dists_notequal))
    avg_equal = np.mean(dists_equal)
    avg_notequal = np.mean(dists_notequal)
    fout.write('Average distance between patches with equal reflectance: {0}\n'.format(avg_equal))
    fout.write('Standard deviation: {0}\n'.format(np.std(dists_equal)))
    fout.write('Average distance between patches with not equal reflectance: {0}\n'.format(avg_notequal))
    fout.write('Standard deviation: {0}\n'.format(np.std(dists_notequal)))
    # Solve for accuracy
    thresacc, acc, thresprec, prec, recall = solve_accuracy(distmetricname, dists_equal, dists_notequal, avg_equal, avg_notequal, stepcount, req_prec)
    fout.write('Best accuracy: {0} at threshold: {1}\n'.format(acc, thresacc))
    fout.write('At ~ {0} precision ({1}) the recall is: {2} at threshold: {3}\n'.format(req_prec, prec, recall, thresprec))

    return thresacc, acc, thresprec, prec, recall


if __name__ == '__main__':
    #test_alexnet()

    with open(os.path.join(ROOTPATH, TESTFILE), 'r') as f:
        lines = f.readlines()

    # separate lines into equal/notequal
    eq_lines = []
    noteq_lines = []
    for l in lines:
        img_path1, img_path2, sim = process_line(l)

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

    featurefilepath = os.path.join(EXPROOTPATH, 'featuredata.dat')
    loaded = False
    if os.path.exists(featurefilepath):
        dataloaded = packer.funpackb_version(1.0, featurefilepath)
        features = dataloaded['features']
        distances_equal = dataloaded['distances_equal']
        distances_notequal = dataloaded['distances_notequal']
        loaded = True
    else:
        features = {}
        distances_equal = {}
        distances_notequal = {}

    # If we preloaded the features, don't compute them again
    # The user can delete the file if he wants to recompute the features
    if not loaded:
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

            #last_grayimg_path = ''
            for l_idx, l in enumerate(progress_bar(sampled_lines)):
                img_path1, img_path2, sim = process_line(l)

                img1 = common.load_image(img_path1, is_srgb=False)
                img2 = common.load_image(img_path2, is_srgb=False)
                # Don't load the image again if it is not necessary
                #if grayimg_path != last_grayimg_path:
                    #head, filenameext = os.path.split(acrp(grayimg_path))
                    #origfilenameext = filenameext.replace('-gray', '-resized')
                    #origfilepath = os.path.join(head, origfilenameext)

                    #if not os.path.exists(origfilepath):
                        #raise ValueError('Image file doesn\'t exist: {0}!'.format(origfilepath))

                    #img = common.load_image(origfilepath, is_srgb=True)
                    #chrom_img = common.compute_chromaticity_image(img)
                    #last_grayimg_path = grayimg_path

                # We list all possible inputs here...
                inputs = {}

                inputs['data'] = img1
                outputblobs1 = comp_feature_func(net, inputs, feature_options, croplen)

                inputs['data'] = img2
                outputblobs2 = comp_feature_func(net, inputs, feature_options, croplen)

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

        # Save data so we don't have to recompute everything again
        datatosave = {}
        datatosave['features'] = features
        datatosave['distances_equal'] = distances_equal
        datatosave['distances_notequal'] = distances_notequal
        packer.fpackb(datatosave, 1.0, featurefilepath)

    scores = []
    indices = []

    fout = open(os.path.join(EXPROOTPATH, 'output.txt'), 'w')

    for net_name, net_options in network_options.iteritems():
        fout.write('############# {0} #############\n'.format(net_name))

        feature_options = net_options['feature_options']
        for f in feature_options:
            for dm_idx, dm in enumerate(distmetrics):
                indices.append((net_name, f, dm['name']))
                score = analyze_distance_metric(
                    '{0}-{1}-{2}'.format(net_name, f, dm['name']),
                    distances_equal[net_name][f][dm_idx],
                    distances_notequal[net_name][f][dm_idx],
                    STEPCOUNT, REQUIREDPRECISION, fout)
                scores.append(score)

    # Search for best results
    fout.write('########### BEST RESULTS ###########\n')
    scores = np.array(scores)
    max_inds = np.argmax(scores, axis=0)
    net_name, f, dmname = indices[max_inds[1]]
    accuracy = scores[max_inds[1], 1]
    fout.write('All values: {0}\n'.format(scores[:, 1]))
    fout.write('Best accuracy {0}: {1} network, {2} feature with {3} distance metric\n'.format(accuracy, net_name, f, dmname))
    net_name, f, dmname = indices[max_inds[4]]
    recall = scores[max_inds[4], 4]
    fout.write('All values: {0}\n'.format(scores[:, 4]))
    fout.write('Best recall {0}: {1} network, {2} feature with {3} distance metric\n'.format(recall, net_name, f, dmname))

    fout.close()

