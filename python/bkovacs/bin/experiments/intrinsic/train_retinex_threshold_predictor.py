import sys
import os
import multiprocessing

import numpy as np
from sklearn.linear_model import LogisticRegression, LinearRegression, RidgeCV, ElasticNetCV, LassoCV
from progressbar import ProgressBar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar, progress_bar_widgets
from lib.utils.data import common, fileproc
from lib.utils.misc import packer
from lib.utils.net.misc import init_net
from lib.utils.train.ml import split_train_test

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe

INPUTSIZE = 224
MODEL_FILE = acrp('models/vgg_cnn_m/VGG_CNN_M_deploy.prototxt')
PRETRAINED_WEIGHTS = acrp('models/vgg_cnn_m/VGG_CNN_M.caffemodel')
DATAROOTPATH = acrp('data/iiw-dataset/data')
TRAINING_FILEPATH = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues.txt')
FEATURES_FILEPATH = acrp('experiments/mitintrinsic/allresults/features.dat')
SCORES_FILEPATHBASE = acrp('experiments/mitintrinsic/allresults/retinex-threshold-training-scores')
MEAN_FILE = acrp('models/vgg_cnn_m/VGG_mean.binaryproto')
USE_SAVED_FEATURES = False


def build_matrices(data_set, datarootpath, features, input_size):
    output_blobs = features

    matrices = {}
    n_samples = len(data_set)
    Xs = []
    ys = []

    for f in features:
        # Get the size of the feature from the network
        feature_blob = net.blobs[f]
        n_features = feature_blob.data.size
        print '{0} feature size: {1}'.format(f, n_features)

        X = np.empty((n_samples, n_features))
        y = np.empty((n_samples))

        Xs.append(X)
        ys.append(y)

    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=n_samples)
    pbar.start()

    for i, (tag, threshold_chrom) in enumerate(data_set.iteritems()):
        # TODO: Linearize image??
        img = common.load_image(os.path.join(DATAROOTPATH, '{0}.png'.format(tag)), is_srgb=False)
        img = common.resize_and_crop_image(img, resize=input_size, crop=None, keep_aspect_ratio=False)

        prediction = compute_feature(net, input_name, img, output_blobs)

        for f_idx, f in enumerate(features):
            blobdata = prediction[f]
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
        #features.append('fc7')
        #features.append('pool5')
        features.append('pool1')

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
            Xys.append(build_matrices(s, DATAROOTPATH, features, INPUTSIZE))

        packer.fpackb({'Xys': Xys, 'splits': splits, 'features': features}, 1.0, FEATURES_FILEPATH)

    n_cpus = multiprocessing.cpu_count() - 1
    models = {}
    cv_folds = n_cpus

    if 'RidgeCV' in usedmodels:
        ridgecv_alphas = np.logspace(-5.0, 10.0, 50)
        print 'Trying alphas for RidgeCV: {0}'.format(ridgecv_alphas)
        models['RidgeCV'] = RidgeCV(alphas=ridgecv_alphas, fit_intercept=True, normalize=False, scoring=None, score_func=None, loss_func=None, cv=cv_folds, gcv_mode=None, store_cv_values=False)

    if 'LassoCV' in usedmodels:
        models['LassoCV'] = LassoCV(eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=5000, tol=0.0001, copy_X=True, cv=cv_folds, verbose=True, n_jobs=n_cpus, positive=False)

    if 'ElasticNetCV' in usedmodels:
        elasticnetcv_l1_ratios = np.linspace(0.001, 0.999, 50)
        print 'Trying l1_ratios for ElasticNetCV: {0}'.format(elasticnetcv_l1_ratios)
        models['ElasticNetCV'] = ElasticNetCV(l1_ratio=elasticnetcv_l1_ratios, eps=0.001, n_alphas=100, alphas=None, fit_intercept=True, normalize=False, precompute='auto', max_iter=5000, tol=0.0001, cv=cv_folds, copy_X=True, verbose=True, n_jobs=n_cpus, positive=False)

    n_features = len(features)
    n_models = len(models)
    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=n_features * n_models)
    pbar_counter = 0
    pbar.start()

    result_data = {f: {} for f in features}
    samplecount = 5000

    for i, feature_name in enumerate(features):
        print 'Training and testing for feature {0}'.format(feature_name)
        for j, (model_name, model) in enumerate(models.iteritems()):
            print 'Using model {0}'.format(model_name)
            result_data[feature_name][model_name] = {}

            # Get matrices for training set
            Xs, ys = Xys[0]
            X = Xs[i]
            y = ys[i]

            #X = X[np.random.choice(X.shape[0], size=samplecount), :]
            #y = y[np.random.choice(y.shape[0], size=samplecount)]
            y_train_mean = np.mean(y)
            model.fit(X, y)
            result_data[feature_name][model_name]['trained_model'] = model

            params = {}
            params['alpha'] = model.alpha_
            if model_name == 'ElasticNetCV':
                params['l1_ratio'] = model.l1_ratio_

            result_data[feature_name][model_name]['best_params'] = params
            print 'Best parameters: {0}'.format(params)

            training_score = model.score(X, y)
            print 'Training score: {0}'.format(training_score)

            # Get matrices for test set
            Xs, ys = Xys[1]
            X = Xs[i]
            y = ys[i]

            y_pred = model.predict(X)
            r2_error = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred) ** 0.5
            mae = mean_absolute_error(y, y_pred)
            result_data[feature_name][model_name]['r2_error'] = r2_error
            result_data[feature_name][model_name]['rmse'] = rmse
            result_data[feature_name][model_name]['mae'] = mae

            print 'Test scores: R2 error {0}, RMSE {1}, mean absolute error {2}'.format(r2_error, rmse, mae)

            y_pred[:] = y_train_mean
            r2_error = r2_score(y, y_pred)
            rmse = mean_squared_error(y, y_pred) ** 0.5
            mae = mean_absolute_error(y, y_pred)

            print 'Test scores if the prediction is the mean of the training set: R2 error {0}, RMSE {1}, mean absolute error {2}'.format(r2_error, rmse, mae)

            pbar_counter += 1
            pbar.update(pbar_counter)

    pbar.finish()

    packer.fpackb({'result_data': result_data}, 1.0, '{0}-{1}.dat'.format(SCORES_FILEPATHBASE, '-'.join(usedmodels)), use_msgpack=False)


