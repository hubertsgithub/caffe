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
TESTFILE = 'alllinks_test.txt'

METADATA_FILE = '/file_categories_unique.npy'
FEATURE_FILE = 'data/features_googlenet-interclasslinks_0-830578.npy'

STEPCOUNT = 500
REQUIREDPRECISION = 0.9


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
    F1scores = []
    for threshold in np.linspace(0, avg_notequal * 3, stepcount):
        confmx = compute_confusionmx(dists_equal, dists_notequal, threshold)
        acc = compute_accuracy(confmx)
        recall = compute_recall(confmx)
        prec = compute_precision(confmx)
        F1 = 2 * prec * recall / (prec + recall)

        accuracies.append([threshold, acc])
        recalls.append([threshold, recall])
        precisions.append([threshold, prec])
        F1scores.append([threshold, F1])

    F1scores = np.array(F1scores)
    accuracies = np.array(accuracies)
    recalls = np.array(recalls)
    precisions = np.array(precisions)
    precrecall = np.empty_like(recalls)
    precrecall[:, 0] = recalls[:, 1]
    precrecall[:, 1] = precisions[:, 1]

    plot_and_save_2D_array(os.path.join(EXPROOTPATH, '{0}-threshold-accuracies.png'.format(distmetricname)), accuracies, xlabel='threshold', ylabel='F1scores', yinterval=(0.0, 1.0))
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

    metadata = np.load(ROOTPATH + METADATA_FILE)
    features = np.load(EXPROOTPATH + FEATURE_FILE)

    with open(os.path.join(ROOTPATH, TESTFILE), 'r') as f:
        lines = f.readlines()

    # separate lines into equal/notequal
    eq_lines_left = []
    noteq_lines_left = []
    eq_lines_right = []
    noteq_lines_right = []
    for l in lines:
        img_path1, img_path2, sim = process_line(l)
        if sim:
            eq_lines_left.append(img_path1)
            eq_lines_right.append(img_path2)
        else:
            noteq_lines_left.append(img_path1)
            noteq_lines_right.append(img_path2)

    # reduce noteq_lines to amount of equal lines
    nums = np.zeros(len(noteq_lines_left))
    nums[:len(eq_lines_left)] = 1
    np.random.shuffle(nums)

    #convert to numpy
    eq_lines_left = np.array(eq_lines_left)
    eq_lines_right = np.array(eq_lines_right)
    noteq_lines_left = np.array(noteq_lines_left)[nums==1]
    noteq_lines_right = np.array(noteq_lines_right)[nums==1]
    print eq_lines_left.shape
    print features[np.in1d(metadata[:,0], eq_lines_left),:].shape
    print features[np.in1d(metadata[:,0], eq_lines_right),:].shape
    # print np.all(eq_lines_left[np.in1d(eq_lines_left, metadata[:,0])] == eq_lines_left)
    # print eq_lines_right[np.in1d(eq_lines_right, metadata[:,0])].shape

    x = np.array([0,1,2,3,4,5,6,7,8,9,10])
    y = np.array([6,7,8,9,10])
    print np.in1d(y,x)
    print np.in1d(x,y)
    print x[np.in1d(x,y)]
    # equal_features = np.zeros((eq_lines_left.shape[0],256,2))
    equal_features_l = features[np.in1d(metadata[:,0], eq_lines_left)]
    equal_features_r = features[np.in1d(metadata[:,0], eq_lines_right)]
    equal_distances = np.linalg.norm(equal_features_l-equal_features_r, axis=1)

    notequal_features_l = features[np.in1d(metadata[:,0], noteq_lines_left)]
    notequal_features_r = features[np.in1d(metadata[:,0], noteq_lines_right)]
    notequal_distances = np.linalg.norm(notequal_features_l-notequal_features_r, axis=1)

    distances_equal = list(equal_distances)
    distances_notequal = list(notequal_distances)

    # noteq_lines = random.sample(noteq_lines, len(eq_lines))

    # eq_lines = random.sample(eq_lines, 100 / 2)
    # noteq_lines = random.sample(noteq_lines, 100 / 2)

    # sampled_lines = eq_lines + noteq_lines
    
    

    # distances_equal = []
    # distances_notequal = []
    # # compute distances for lines
    # for l_idx, l in enumerate(progress_bar(sampled_lines)):
    #     img_path1, img_path2, sim = process_line(l)
    #     feat_1 = features[np.where(metadata[:,0]==img_path1)]
    #     feat_2 = features[np.where(metadata[:,0]==img_path2)]
    #     dist = np.linalg.norm(feat_1 - feat_2)
    #     if sim:
    #         distances_equal.append(dist)
    #     else:
    #         distances_notequal.append(dist)

    fout = open(os.path.join(EXPROOTPATH, 'output.txt'), 'w')

    score = analyze_distance_metric(
        'test_experiment',
        distances_equal,
        distances_notequal,
        STEPCOUNT, REQUIREDPRECISION, fout)

    fout.close()

