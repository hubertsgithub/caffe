import os
import numpy as np

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc import packer
from lib.intrinsic import intrinsic

ROOTPATH = acrp('experiments/mitintrinsic/allresults')
SCORE_FILENAME = 'ALLDATA-dense-oraclethreshold.dat'
SCORE_FILENAME2 = 'ALLDATA-exceptdense-oraclethreshold.dat'
OUTPUT_FILENAME = 'best-thresholdvalues.txt'
ESTIMATORCLASSSTR = 'lib.intrinsic.intrinsic.Zhao2012GroundTruthGroupsEstimator'
PARAMNAME = 'threshold_chrom'
IIWTAGPATH = acrp('data/iiw-dataset/denseimages.txt')
IIWTAGPATH2 = acrp('data/iiw-dataset/all-except-denseimages.txt')


def extract_thresholds(tagfile_path, scorefile_path):
    with open(tagfile_path) as f:
        tags = [s.strip() for s in f.readlines()]

    dic = packer.funpackb_version(1.1, scorefile_path)
    allscores = dic['allscores']
    if ESTIMATORCLASSSTR not in allscores:
        raise ValueError('Can\'t find {0} among the evaluated scores'.format(ESTIMATORCLASSSTR))

    scores = allscores[ESTIMATORCLASSSTR]
    choices = intrinsic.Zhao2012GroundTruthGroupsEstimator.param_choices()

    if len(tags) != scores.shape[0]:
        raise ValueError('The number of tags should be equal to the first dimension of shapes: {0} tags != {1} scores.shape[0]'.format(len(tags), scores.shape[0]))

    if len(choices) != scores.shape[1]:
        raise ValueError('The number of choices should be equal to the second dimension of shapes: {0} choices != {1} scores.shape[1]'.format(len(tags), scores.shape[1]))

    ret = []
    for i, tag in enumerate(tags):
        # Get the best parameter configuration
        best_choice = np.argmin(scores[i, :])

        bestparams = choices[best_choice]
        score = scores[i, best_choice]
        print 'tag: {0}, bestparams: {1}, score: {2}'.format(tag, bestparams, score)
        ret.append('{0} {1} {2}\n'.format(tag, bestparams[PARAMNAME], score))

    return ret


if __name__ == '__main__':
    scorefile_path = os.path.join(ROOTPATH, SCORE_FILENAME)
    lines = extract_thresholds(IIWTAGPATH, scorefile_path)

    scorefile_path2 = os.path.join(ROOTPATH, SCORE_FILENAME2)
    lines2 = extract_thresholds(IIWTAGPATH2, scorefile_path2)

    lines = lines + lines2

    fout = open(os.path.join(ROOTPATH, OUTPUT_FILENAME), 'w')
    for l in lines:
        fout.write(l)

    fout.close()

