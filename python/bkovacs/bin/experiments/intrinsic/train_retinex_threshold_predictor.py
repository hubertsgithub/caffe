import sys
import os
import multiprocessing

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, ElasticNetCV, LassoCV
from progressbar import ProgressBar

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar, progress_bar_widgets
from lib.utils.data import common, fileproc
from lib.utils.misc import packer
from lib.utils.net.misc import init_net
from lib.utils.train.ml import split_train_test

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe

MODEL_FILE = acrp('models/vgg_cnn_m/VGG_CNN_M_deploy.prototxt')
PRETRAINED_WEIGHTS = acrp('models/vgg_cnn_m/VGG_CNN_M.caffemodel')
DATAROOTPATH = acrp('data/iiw-dataset/data')
TRAINING_FILEPATH = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues.txt')
FEATURES_FILEPATH = acrp('experiments/mitintrinsic/allresults/features.dat')
SCORES_FILEPATHBASE = acrp('experiments/mitintrinsic/allresults/retinex-threshold-training-scores')
MEAN_FILE = acrp('models/vgg_cnn_m/VGG_mean.binaryproto')
USE_SAVED_FEATURES = True


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
    possiblemodels = ['RidgeCV', 'LassoCV', 'ElasticNetCV']
    usedmodels = sys.argv[1:]
    if len(usedmodels) == 0 or not np.all([x in possiblemodels for x in usedmodels]):
        print 'The possible models are: {0}'.format(possiblemodels)
        sys.exit(1)

    print 'Using models: {0}'.format(usedmodels)

    lines = fileproc.freadlines(TRAINING_FILEPATH)

    best_thresholds = {}
    for l in lines:
        tokens = l.split(' ')
        tag, threshold_chrom, score = tokens

        best_thresholds[tag] = threshold_chrom

    if USE_SAVED_FEATURES:
        print 'Reading computed features from {0}'.format(FEATURES_FILEPATH)
        dic = packer.funpackb_version(1.0, FEATURES_FILEPATH)
        Xys = dic['Xys']
        splits = dic['splits']
        features = dic['features']
    else:
        features = []
        features.append({'blobname': 'fc7'})
        features.append({'blobname': 'pool5'})

        print 'Computing features for {0} images and {1} different feature configurations'.format(len(best_thresholds), len(features))

        # split into training and testsets
        # 0: train, 1: val, 2: test
        splits = split_train_test(best_thresholds, 0.2, 0.2)
        s = ' '.join([str(len(split)) for split in splits])
        print 'Final sizes: {0}'.format(s)

        input_name = 'data'
        net = init_net(MODEL_FILE, PRETRAINED_WEIGHTS, MEAN_FILE, input_name)

        # Build training matrices: X, y
        Xys = []
        for s in splits:
            Xys.append(build_matrices(s, DATAROOTPATH, features))

    packer.fpackb({'Xys': Xys, 'splits': splits, 'features': features}, 1.0, FEATURES_FILEPATH)

    n_cpus = multiprocessing.cpu_count() - 1
    models = {}

    if 'RidgeCV' in usedmodels:
        ridgecv_alphas = np.logspace(0.0, 1.0, 5)
        print 'Trying alphas for RidgeCV: {0}'.format(ridgecv_alphas)
        models['RidgeCV'] = RidgeCV(alphas=ridgecv_alphas, fit_intercept=True, normalize=False, scoring=None, score_func=None, loss_func=None, cv=None, gcv_mode=None, store_cv_values=False)

    if 'LassoCV' in usedmodels:
        models['LassoCV'] = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=5000, tol=0.0001, copy_X=True, cv=None, verbose=True, n_jobs=n_cpus, positive=False)

    if 'ElasticNetCV' is usedmodels:
        elasticnetcv_l1_ratios = np.linspace(0.1, 0.9, 5)
        print 'Trying l1_ratios for ElasticNetCV: {0}'.format(elasticnetcv_l1_ratios)
        models['ElasticNetCV'] = ElasticNetCV(l1_ratio=elasticnetcv_l1_ratios, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=5000, tol=0.0001, cv=None, copy_X=True, verbose=True, n_jobs=n_cpus, positive=False)

    n_features = len(features)
    n_models = len(models)
    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=n_features * n_models)
    pbar_counter = 0
    pbar.start()

    scores = [{} for _ in xrange(n_features)]
    best_params = [{} for _ in xrange(n_features)]
    samplecount = 500

    for i in range(n_features):
        print 'Training and testing for feature {0}'.format(features[i]['blobname'])
        for j, (modelname, model) in enumerate(models.iteritems()):
            print 'Using model {0}'.format(modelname)

            # Get matrices for training set
            Xs, ys = Xys[0]
            X = Xs[i]
            y = ys[i]

            X = X[np.random.choice(X.shape[0], size=samplecount), :]
            y = y[np.random.choice(y.shape[0], size=samplecount)]
            model.fit(X, y)
            print 'Best parameters: {0}'.format(model.get_params())
            best_params[i][modelname] = model.get_params()

            training_score = model.score(X, y)
            print 'Training score: {0}'.format(training_score)

            # Get matrices for test set
            Xs, ys = Xys[1]
            X = Xs[i]
            y = ys[i]
            test_score = model.score(X, y)
            print 'Test score: {0}'.format(test_score)
            scores[i][modelname] = test_score

            pbar_counter += 1
            pbar.update(pbar_counter)

    pbar.finish()

    packer.fpackb({'scores': scores, 'best_params': best_params}, 1.0, '{0}-{1}.dat'.format(SCORES_FILEPATHBASE, '-'.join(usedmodels)))


