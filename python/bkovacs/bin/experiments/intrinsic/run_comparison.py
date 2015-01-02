import os
import sys
import random

from lib.intrinsic import intrinsic
from lib.intrinsic import comparison

from lib.utils.misc.pathresolver import acrp

# 0 mitintrinsic
# 1 Sean's synthetic dataset
# 2 IIW dense
DATASETCHOICE = 2

SAVEROOTDIR = acrp('experiments/mitintrinsic/allresults')
IIWTAGPATH = acrp('data/iiw-dataset/denseimages.txt')

# The following objects were used in the evaluation. For the learning algorithms
# (not included here), we used two-fold cross-validation with the following
# randomly chosen split.
SET1MIT = ['box', 'cup1', 'cup2', 'dinosaur', 'panther', 'squirrel', 'sun', 'teabag2']
SET2MIT = ['deer', 'frog1', 'frog2', 'paper1', 'paper2', 'raccoon', 'teabag1', 'turtle']

SETINDOOR = map(lambda n: str(n), range(1, 25))

random.seed(10)
with open(IIWTAGPATH) as f:
    SETIIWDENSE = [s.strip() for s in f.readlines()][:10]

if DATASETCHOICE == 0:
    ALL_TAGS = SET1MIT + SET2MIT
    ERRORMETRIC = 0  # LMSE
elif DATASETCHOICE == 1:
    ALL_TAGS = SETINDOOR
    ERRORMETRIC = 0  # LMSE
elif DATASETCHOICE == 2:
    ALL_TAGS = SETIIWDENSE
    ERRORMETRIC = 1  # WHDR
else:
    raise ValueError('Unknown dataset choice: {0}'.format(DATASETCHOICE))

# The following four objects weren't used in the evaluation because they have
# slight problems, but you may still find them useful.
EXTRA_TAGS = ['apple', 'pear', 'phone', 'potato']

# Use L1 to compute the final results. (For efficiency, the parameters are still
# tuned using least squares.)
USE_L1 = False

# Output of the algorithms will be saved here
if USE_L1:
    RESULTS_DIR = os.path.join(SAVEROOTDIR, 'results_L1')
else:
    RESULTS_DIR = os.path.join(SAVEROOTDIR, 'results')

ESTIMATORS = [
                ('Baseline (BAS)', intrinsic.BaselineEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using RGB images', intrinsic.GrayscaleRetinexWithThresholdImageRGBEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, small network 3 conv layers', intrinsic.GrayscaleRetinexWithThresholdImageChromSmallNetEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers, concatenated conv1+3 output', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetConcatEstimator),
                #('Grayscale Retinex with CNN predicted threshold images using chromaticity + grayscale image, big network 4 conv layers, concatenated conv1+3 output + maxpool between conv1-2 and 2-3', intrinsic.GrayscaleRetinexWithThresholdImageChromBigNetConcatMaxpoolEstimator),
                #('Grayscale Retinex with ground truth threshold images', intrinsic.GrayscaleRetinexWithThresholdImageGroundTruthEstimator),
                ('Zhao2012', intrinsic.Zhao2012Estimator),
                ('Zhao2012 with ground truth reflectance groups', intrinsic.Zhao2012GroundTruthGroupsEstimator),
                ('Grayscale Retinex (GR-RET)', intrinsic.GrayscaleRetinexEstimator),
                ('Color Retinex (COL-RET)', intrinsic.ColorRetinexEstimator),
                #("Weiss's Algorithm (W)", intrinsic.WeissEstimator),
                #('Weiss + Retinex (W+RET)', intrinsic.WeissRetinexEstimator),
                ]


if __name__ == '__main__':
    if len(sys.argv) != 2 or sys.argv[1] not in ['dispatch', 'aggregate']:
        print 'Usage: run_comparison.py dispatch|aggregate'
        sys.exit(1)

    option = sys.argv[1]
    if option == 'dispatch':
        comparison.dispatch_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS)
    else:
        comparison.aggregate_comparison_experiment(DATASETCHOICE, ALL_TAGS, ERRORMETRIC, USE_L1, RESULTS_DIR, ESTIMATORS)

