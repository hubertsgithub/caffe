import os
import numpy as np
from progressbar import ProgressBar
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from lib.utils.data import fileproc, common
from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar, progress_bar_widgets
from lib.utils.net.misc import init_net


def compute_prediction(net, inputs, output_name):
    conv_inputs = {}
    for input_name, input_image in inputs.iteritems():
        caffe_in = net.preprocess(input_name, input_image)
        conv_inputs[input_name] = caffe_in[np.newaxis, ...]

    # TODO: np.newaxis??
    #prediction = net.forward_all(blobs=[], **{input_name: caffe_in[np.newaxis, :, :, :]})
    prediction = net.forward_all(blobs=[], **conv_inputs)

    return prediction[output_name]


def extract_ys(experiment_options):
    lines = fileproc.freadlines(experiment_options['train_filepath'])
    y_train = np.empty((len(lines)))
    for l_idx, l in enumerate(lines):
        tokens = l.split(' ')
        gray_file, chrom_file, threshold_chrom = tokens
        y_train[l_idx] = threshold_chrom

    y_train_mean = np.mean(y_train)

    lines = fileproc.freadlines(experiment_options['test_filepath'])

    net = init_net(experiment_options['model_file'], \
                   experiment_options['pretrained_weights'], \
                   experiment_options['mean_val'], \
                   experiment_options['input_config'])

    best_thresholds = {}
    y = np.empty((len(lines)))
    y_pred = np.empty((len(lines)))

    for l_idx, l in enumerate(progress_bar(lines)):
        tokens = l.split(' ')
        gray_file, chrom_file, threshold_chrom = tokens
        y[l_idx] = threshold_chrom

        # Linearize image
        gray_img = common.load_image(acrp(gray_file), is_srgb=False)
        gray_img = common.resize_and_crop_image(gray_img, experiment_options['input_size'], crop=None, keep_aspect_ratio=False)
        gray_img = gray_img ** 2.2
        chrom_img = common.load_image(acrp(chrom_file), is_srgb=False)
        chrom_img = common.resize_and_crop_image(chrom_img, experiment_options['input_size'], crop=None, keep_aspect_ratio=False)
        chrom_img = chrom_img ** 2.2
        inputs = {'data': gray_img[:, :, np.newaxis], 'chrom': chrom_img}

        y_pred[l_idx] = compute_prediction(net, inputs, experiment_options['output_name'])
        print y_pred[l_idx]

    return y_train_mean, y, y_pred


if __name__ == '__main__':
    experiments = []
    common_config = {'mean_val': 128, \
                     'input_config': {'data': {'raw_scale': 255, 'input_scale': 1.0/255}, \
                                      'chrom': {'channel_swap': (2, 1, 0), 'raw_scale': 255, 'input_scale': 1.0/255}}, \
                     'input_size': 256}

    experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-dense.txt'), \
                        'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
                        'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2fc.prototxt'), \
                        'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_2fc_iter_30000.caffemodel'), \
                        'output_name': 'fc3'}.items() + common_config.items()))
    experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-dense.txt'), \
                        'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
                        'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2fc.prototxt'), \
                        'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_2fc_iter_100000.caffemodel'), \
                        'output_name': 'fc3'}.items() + common_config.items()))
    experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-all.txt'), \
                        'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
                        'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2fc.prototxt'), \
                        'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_bigdata_2fc_iter_30000.caffemodel'), \
                        'output_name': 'fc3'}.items() + common_config.items()))
    #experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-all.txt'), \
    #                    'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
    #                    'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2fc.prototxt'), \
    #                    'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_bigdata_2fc_iter_100000.caffemodel'), \
    #                    'output_name': 'fc3'}.items() + common_config.items()))
    experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-dense.txt'), \
                        'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
                        'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2conv_2fc.prototxt'), \
                        'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_2conv_2fc_iter_30000.caffemodel'), \
                        'output_name': 'fc4'}.items() + common_config.items()))
    experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-dense.txt'), \
                        'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
                        'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2conv_2fc.prototxt'), \
                        'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_2conv_2fc_iter_100000.caffemodel'), \
                        'output_name': 'fc4'}.items() + common_config.items()))
    #experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-all.txt'), \
    #                    'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
    #                    'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2conv_2fc.prototxt'), \
    #                    'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_bigdata_2conv_2fc_iter_30000.caffemodel'), \
    #                    'output_name': 'fc4'}.items() + common_config.items()))
    #experiments.append(dict({'train_filepath': acrp('data/iiw-dataset/train-threshold-all.txt'), \
    #                    'test_filepath': acrp('data/iiw-dataset/test-threshold-all.txt'), \
    #                    'model_file': acrp('ownmodels/mitintrinsic/deploy_iiw_thresholdpredictor_2conv_2fc.prototxt'), \
    #                    'pretrained_weights': acrp('ownmodels/mitintrinsic/snapshots/caffenet_train_iiw_thresholdpredictor_bigdata_2conv_2fc_iter_100000.caffemodel'), \
    #                    'output_name': 'fc4'}.items() + common_config.items()))

    results = {}
    for e in experiments:
        y_train_mean, y, y_pred = extract_ys(e)

        r2_error = r2_score(y, y_pred)
        rmse = mean_squared_error(y, y_pred) ** 0.5
        mae = mean_absolute_error(y, y_pred)

        y_pred[:] = y_train_mean
        r2_error_tm = r2_score(y, y_pred)
        rmse_tm = mean_squared_error(y, y_pred) ** 0.5
        mae_tm = mean_absolute_error(y, y_pred)

        results[e['pretrained_weights']] = (r2_error, rmse, mae, r2_error_tm, rmse_tm, mae_tm)

    for name, errors in results.iteritems():
        r2_error, rmse, mae, r2_error_tm, rmse_tm, mae_tm = errors
        print 'Weights: {0}'.format(name)

        print 'Test scores: R2 error {0}, RMSE {1}, mean absolute error {2}'.format(r2_error, rmse, mae)
        print 'Test scores if the prediction is the mean of the training set: R2 error {0}, RMSE {1}, mean absolute error {2}'.format(r2_error_tm, rmse_tm, mae_tm)

