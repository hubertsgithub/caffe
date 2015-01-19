import math
import os
import random

import numpy as np

from lib.utils.data import fileproc
from lib.utils.misc.pathresolver import acrp
from lib.utils.train import ml

DATAPATH = 'data/iiw-dataset/data'

THRESHOLD_FILEPATH_ALL = acrp('experiments/mitintrinsic/allresults/best-thresholdvalues-all.txt')
TRAIN_FILEPATH_ALL = acrp('data/iiw-dataset/train-bin-threshold-all.txt')
TEST_FILEPATH_ALL = acrp('data/iiw-dataset/test-bin-threshold-all.txt')
JOBS_FILEPATH_ALL = acrp('data/iiw-dataset/jobs-bin-threshold-all.txt')

N_lo = 0.75
N_hi = 2.
P_hi = 0.1
STEPS = 10
# We should have equal number from all classes!
RANDOMSAMPLES = 5


def save_train_test_files(threshold_filepath, train_filepath, test_filepath, jobs_filepath):
    print 'Creating train and test files: {0}, {1}'.format(train_filepath, test_filepath)
    lines = fileproc.freadlines(threshold_filepath)

    # Select 20% test set randomly
    train_set_lines, test_set_lines = ml.split_train_test(lines, 0.2)
    train_set = []
    test_set = []
    job_lines = []

    for lines, conv_lines in [(train_set_lines, train_set), (test_set_lines, test_set)]:
        for l in lines:
            tokens = l.split(' ')
            tag, threshold_chrom, score = tokens

            # compute the logarithm of the threshold_chrom and try to predict that!
            threshold_chrom = math.log(float(threshold_chrom))
            sample_intervals = []
            # greater
            sample_intervals.append([N_lo, N_hi])
            # less
            sample_intervals.append([-N_hi, -N_lo])
            # approx equal
            sample_intervals.append([-P_hi, P_hi])

            for class_idx, (lo, hi) in enumerate(sample_intervals):
                samples = random.sample(np.linspace(lo, hi, STEPS), RANDOMSAMPLES)

                for sample_idx, shift in enumerate(samples):
                    # distorted threshold
                    shifted_threshold_chrom = threshold_chrom + shift

                    shading_file = '{0}-classnum{1}-samplenum{2}-shading.png'.format(tag, class_idx, sample_idx)
                    shading_file = os.path.join(DATAPATH, tag, shading_file)
                    refl_file = '{0}-classnum{1}-samplenum{2}-refl.png'.format(tag, class_idx, sample_idx)
                    refl_file = os.path.join(DATAPATH, tag, refl_file)
                    conv_line = '{0} {1} {2}'.format(shading_file, refl_file, class_idx)
                    job_line = '{0} {1} {2} {3} {4}'.format(tag, threshold_chrom, class_idx, sample_idx, shift)

                    conv_lines.append(conv_line)
                    job_lines.append(job_line)

    print 'Training set ({0} items), test set ({1} items) created'.format(len(train_set), len(test_set))

    fileproc.fwritelines(train_filepath, train_set)
    fileproc.fwritelines(test_filepath, test_set)
    fileproc.fwritelines(jobs_filepath, job_lines)


if __name__ == "__main__":
    file_options = []
    file_options.append([THRESHOLD_FILEPATH_ALL, TRAIN_FILEPATH_ALL, TEST_FILEPATH_ALL, JOBS_FILEPATH_ALL])

    for threshold_filepath, train_filepath, test_filepath, use_orig_file in file_options:
        save_train_test_files(threshold_filepath, train_filepath, test_filepath, use_orig_file)

    print 'Done.'

