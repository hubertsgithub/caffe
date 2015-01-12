import os

from lib.utils.data import fileproc
from lib.utils.train import ml
from lib.utils.misc.pathresolver import acrp

DATAPATH = 'data/iiw-dataset/data'
THRESHOLD_FILEPATH_DENSE = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues-dense.txt')
TRAIN_FILEPATH_DENSE = acrp('data/iiw-dataset/train-threshold-dense.txt')
TEST_FILEPATH_DENSE = acrp('data/iiw-dataset/test-threshold-dense.txt')
TRAIN_FILEPATH_DENSE_RGB = acrp('data/iiw-dataset/train-threshold-rgb-dense.txt')
TEST_FILEPATH_DENSE_RGB = acrp('data/iiw-dataset/test-threshold-rgb-dense.txt')

THRESHOLD_FILEPATH_ALL = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues-all.txt')
TRAIN_FILEPATH_ALL = acrp('data/iiw-dataset/train-threshold-all.txt')
TEST_FILEPATH_ALL = acrp('data/iiw-dataset/test-threshold-all.txt')
TRAIN_FILEPATH_ALL_RGB = acrp('data/iiw-dataset/train-threshold-rgb-all.txt')
TEST_FILEPATH_ALL_RGB = acrp('data/iiw-dataset/test-threshold-rgb-all.txt')


def save_train_test_files(threshold_filepath, train_filepath, test_filepath, use_orig_file):
    print 'Creating train and test files: {0}, {1}'.format(train_filepath, test_filepath)
    lines = fileproc.freadlines(threshold_filepath)

    conv_lines = []
    for l in lines:
        tokens = l.split(' ')
        tag, threshold_chrom, score = tokens

        if use_orig_file:
            orig_file = '{0}.png'.format(tag)
            orig_file = os.path.join(DATAPATH, orig_file)
            conv_line = '{0} {1}'.format(orig_file, threshold_chrom)
        else:
            gray_file = '{0}-gray.png'.format(tag)
            gray_file = os.path.join(DATAPATH, gray_file)
            chrom_file = '{0}-chrom.png'.format(tag)
            chrom_file = os.path.join(DATAPATH, chrom_file)
            conv_line = '{0} {1} {2}'.format(gray_file, chrom_file, threshold_chrom)

        conv_lines.append(conv_line)

    # Select 20% test set randomly
    train_set, test_set = ml.split_train_test(conv_lines, 0.2)
    print 'Training set ({0} items), test set ({1} items) created'.format(len(train_set), len(test_set))

    fileproc.fwritelines(train_filepath, train_set)
    fileproc.fwritelines(test_filepath, test_set)


if __name__ == "__main__":
    file_options = []
    file_options.append([THRESHOLD_FILEPATH_DENSE, TRAIN_FILEPATH_DENSE, TEST_FILEPATH_DENSE, False])
    file_options.append([THRESHOLD_FILEPATH_DENSE, TRAIN_FILEPATH_DENSE_RGB, TEST_FILEPATH_DENSE_RGB, True])
    file_options.append([THRESHOLD_FILEPATH_ALL, TRAIN_FILEPATH_ALL, TEST_FILEPATH_ALL, False])
    file_options.append([THRESHOLD_FILEPATH_ALL, TRAIN_FILEPATH_ALL_RGB, TEST_FILEPATH_ALL_RGB, True])

    for threshold_filepath, train_filepath, test_filepath, use_orig_file in file_options:
        save_train_test_files(threshold_filepath, train_filepath, test_filepath, use_orig_file)

    print 'Done.'


