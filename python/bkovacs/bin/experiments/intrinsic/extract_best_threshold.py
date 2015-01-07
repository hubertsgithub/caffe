import os
import numpy as np

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc import packer
from lib.intrinsic import intrinsic

ROOTPATH = acrp('experiments/mitintrinsic/allresults')
SCORE_FILENAME = 'ALLDATA-dense-oraclethreshold.dat'
OUTPUT_FILENAME = 'dense-best-thresholdvalues.txt'
ESTIMATORCLASSSTR = 'lib.intrinsic.intrinsic.Zhao2012GroundTruthGroupsEstimator'
PARAMNAME = 'threshold_chrom'
IIWTAGPATH = acrp('data/iiw-dataset/denseimages.txt')
IIWTAGPATH2 = acrp('data/iiw-dataset/all-except-denseimages.txt')

if __name__ == '__main__':
    with open(IIWTAGPATH) as f:
        SETIIWDENSE = [s.strip() for s in f.readlines()]

    tags = SETIIWDENSE

    dic = packer.funpackb_version(1.1, os.path.join(ROOTPATH, SCORE_FILENAME))
    allscores = dic['allscores']
    if ESTIMATORCLASSSTR not in allscores:
        raise ValueError('Can\'t find {0} among the evaluated scores'.format(ESTIMATORCLASSSTR))

    scores = allscores[ESTIMATORCLASSSTR]
    choices = intrinsic.Zhao2012GroundTruthGroupsEstimator.param_choices()

    assert len(tags) == scores.shape[0]
    assert len(choices) == scores.shape[1]

    fout = open(os.path.join(ROOTPATH, OUTPUT_FILENAME), 'w')
    for i, tag in enumerate(tags):
        # Get the best parameter configuration
        best_choice = np.argmin(scores[i, :])

        bestparams = choices[best_choice]
        score = scores[i, best_choice]
        print 'tag: {0}, bestparams: {1}, score: {2}'.format(tag, bestparams, score)
        fout.write('{0} {1} {2}\n'.format(tag, bestparams[PARAMNAME], score))

    fout.close()

