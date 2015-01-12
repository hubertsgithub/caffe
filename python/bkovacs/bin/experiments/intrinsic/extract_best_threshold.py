import os
import numpy as np

from lib.utils.misc.pathresolver import acrp
from lib.utils.misc import packer
import lib.utils.data.fileproc as fileproc
from lib.intrinsic import intrinsic
from lib.intrinsic.intrinsic import Zhao2012Estimator

ROOTPATH = acrp('experiments/mitintrinsic/allresults')
PARAMNAME = 'threshold_chrom'
OUTPUT_FILENAME_DENSE = 'best-thresholdvalues-dense.txt'
OUTPUT_FILENAME_ALL = 'best-thresholdvalues-all.txt'

#SCORE_FILENAME = 'ALLDATA-dense-oraclethreshold.dat'
#SCORE_FILENAME2 = 'ALLDATA-exceptdense-oraclethreshold.dat'
#IIWTAGPATH = acrp('data/iiw-dataset/denseimages.txt')
#IIWTAGPATH2 = acrp('data/iiw-dataset/all-except-denseimages.txt')
#ESTIMATORCLASSSTR = 'lib.intrinsic.intrinsic.Zhao2012GroundTruthGroupsEstimator'

SCORE_FILENAME = 'ALLDATA-zhao2012-withoutgroups-oraclethreshold.dat'
IIWTAGPATH = acrp('data/iiw-dataset/all-images.txt')
IIWTAGFILTERPATH = acrp('data/iiw-dataset/denseimages.txt')
ESTIMATORCLASS = Zhao2012Estimator


def extract_thresholds(tagfile_path, tagfilterfile_path, scorefile_path, EstimatorClass):
    tags = fileproc.freadlines(tagfile_path)
    if tagfilterfile_path:
        filtertags = fileproc.freadlines(tagfilterfile_path)
    else:
        filtertags = tags

    dic = packer.funpackb_version(1.1, scorefile_path)
    allscores = dic['allscores']
    if str(EstimatorClass) not in allscores:
        raise ValueError('Can\'t find {0} among the evaluated scores'.format(EstimatorClass))

    scores = allscores[str(EstimatorClass)]
    choices = EstimatorClass.param_choices()

    if len(tags) != scores.shape[0]:
        raise ValueError('The number of tags should be equal to the first dimension of shapes: {0} tags != {1} scores.shape[0]'.format(len(tags), scores.shape[0]))

    if len(choices) != scores.shape[1]:
        raise ValueError('The number of choices should be equal to the second dimension of shapes: {0} choices != {1} scores.shape[1]'.format(len(tags), scores.shape[1]))

    ret = []
    for i, tag in enumerate(tags):
        if tag not in filtertags:
            continue

        # Get the best parameter configuration
        best_choice = np.argmin(scores[i, :])

        bestparams = choices[best_choice]
        score = scores[i, best_choice]
        print 'tag: {0}, bestparams: {1}, score: {2}'.format(tag, bestparams, score)
        ret.append('{0} {1} {2}\n'.format(tag, bestparams[PARAMNAME], score))

    return ret


if __name__ == '__main__':
    scorefile_path = os.path.join(ROOTPATH, SCORE_FILENAME)
    lines = extract_thresholds(IIWTAGPATH, IIWTAGFILTERPATH, scorefile_path, ESTIMATORCLASS)
    fileproc.fwritelines(os.path.join(ROOTPATH, OUTPUT_FILENAME_DENSE), lines, endline=False)
    print 'Saved best threshold values for {0} images'.format(len(lines))

    lines = extract_thresholds(IIWTAGPATH, None, scorefile_path, ESTIMATORCLASS)
    fileproc.fwritelines(os.path.join(ROOTPATH, OUTPUT_FILENAME_ALL), lines, endline=False)
    print 'Saved best threshold values for {0} images'.format(len(lines))

