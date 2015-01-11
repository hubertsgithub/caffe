import os

from lib.utils.data import fileproc
from lib.utils.train import ml
from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.parallel import call_with_multiprocessing_pool

DATAPATH = 'data/iiw-dataset/data'
THRESHOLD_FILEPATH = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues.txt')
TRAIN_FILEPATH = acrp('data/iiw-dataset/train-threshold.txt')
TEST_FILEPATH = acrp('data/iiw-dataset/test-threshold.txt')


if __name__ == "__main__":
    lines = fileproc.freadlines(THRESHOLD_FILEPATH)

    conv_lines = []
    for l in lines:
        tokens = l.split(' ')
        tag, threshold_chrom, score = tokens

        gray_file = '{0}-gray.png'.format(tag)
        gray_file = os.path.join(DATAPATH, gray_file)
        chrom_file = '{0}-chrom.png'.format(tag)
        chrom_file = os.path.join(DATAPATH, chrom_file)
        conv_line = '{0} {1} {2}'.format(gray_file, chrom_file, threshold_chrom)
        conv_lines.append(conv_line)

    # Select 20% test set randomly
    train_set, test_set = ml.split_train_test(conv_lines, 0.2)

    fileproc.fwritelines(TRAIN_FILEPATH, train_set)
    fileproc.fwritelines(TEST_FILEPATH, test_set)

    print 'Done.'


