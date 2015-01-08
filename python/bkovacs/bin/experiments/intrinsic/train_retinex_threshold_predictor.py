import sys
import os

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression
from progressbar import ProgressBar

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar, progress_bar_widgets
from lib.utils.data import common, fileproc
from lib.utils.misc import packer
from lib.utils.net.misc import init_net
from lib.utils.train.ml import split_train_val_test

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe

MODEL_FILE = acrp('models/vgg_cnn_m/VGG_CNN_M_deploy.prototxt')
PRETRAINED_WEIGHTS = acrp('models/vgg_cnn_m/VGG_CNN_M.caffemodel')
DATAROOTPATH = acrp('data/iiw-dataset/data')
TRAINING_FILEPATH = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues.txt')
FEATURES_FILEPATH = acrp('experiments/mitintrinsic/allresults/features.dat')
MEAN_FILE = acrp('models/vgg_cnn_m/VGG_mean.binaryproto')


def build_matrices(data_set, datarootpath, features):
    output_blobs = [f['blobname'] for f in features]

    matrices = {}
    n_samples = len(data_set)
    Xs = []
    ys = []

    for f in features:
        # Get the size of the feature from the network
        feature_blob = net.blobs[f['blobname']]
        n_features = feature_blob.data.size
        print '{0} feature size: {1}'.format(f['blobname'], n_features)

        X = np.empty((n_samples, n_features))
        y = np.empty((n_samples))

        Xs.append(X)
        ys.append(y)

    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=n_samples)
    pbar.start()

    for i, (tag, threshold_chrom) in enumerate(data_set.iteritems()):
        # TODO: Linearize image??
        img = common.load_image(os.path.join(DATAROOTPATH, '{0}.png'.format(tag)), is_srgb=False)

        prediction = compute_feature(net, input_name, img, output_blobs)

        for f_idx, f in enumerate(features):
            blobdata = prediction[f['blobname']]
            blobdata = np.ravel(np.squeeze(blobdata))

            Xs[f_idx][i, :] = blobdata
            ys[f_idx][i] = threshold_chrom

        pbar.update(i)

    pbar.finish()

    return Xs, ys


def compute_feature(net, input_name, input_image, output_blobs):
    caffe_in = net.preprocess(input_name, input_image)
    prediction = net.forward_all(blobs=output_blobs, **{input_name: caffe_in[np.newaxis, :, :, :]})

    return prediction


if __name__ == '__main__':
    #test_alexnet()
    input_name = 'data'
    net = init_net(MODEL_FILE, PRETRAINED_WEIGHTS, MEAN_FILE, input_name)

    lines = fileproc.freadlines(TRAINING_FILEPATH)

    best_thresholds = {}
    for l in lines:
        tokens = l.split(' ')
        tag, threshold_chrom, score = tokens

        best_thresholds[tag] = threshold_chrom

    # split into training and testsets
    # 0: train, 1: val, 2: test
    splits = split_train_val_test(best_thresholds, 0.2, 0.2)
    s = ' '.join([str(len(split)) for split in splits])
    print 'Final sizes: {0}'.format(s)

    features = []
    features.append({'blobname': 'fc7'})
    features.append({'blobname': 'pool5'})

    #train_set, _ = split_train_test(train_set, 0.9)
    # Build training matrices: X, y
    Xys = []
    for s in splits:
        Xys.append(build_matrices(s, DATAROOTPATH, features))

    packer.fpackb({'Xys': Xys, 'splits': splits}, 1.0, FEATURES_FILEPATH)
    model = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
    #model = LogisticRegression(penalty='l2', dual=False, tol=0.0001, C=1.0, fit_intercept=True, intercept_scaling=1, class_weight=None, random_state=None)

    for i in range(len(features)):
        # Get matrices for training set
        Xs, ys = Xys[0]
        X = Xs[i]
        y = ys[i]
        model.fit(X, y)

        training_score = model.score(X, y)
        print 'Training score: {0}'.format(training_score)

        # Get matrices for test set
        Xs, ys = Xys[2]
        X = Xs[i]
        y = ys[i]
        test_score = model.score(X, y)
        print 'Test score: {0}'.format(test_score)






