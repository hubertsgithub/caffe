import random

import numpy as np

from lib.utils.misc.pathresolver import acrp
from lib.utils.data import fileproc
from lib.utils.misc.progressbaraux import progress_bar

TRAINFILEPATH_PATCH = acrp('data/iiw-dataset/train.txt')
TRAINNEGSFILEPATH_PATCH = acrp('data/iiw-dataset/train-negs.txt')

TRAINFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/train-rgb.txt')
TRAINNEGSFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/train-rgb-negs.txt')
NEGS_MULTI = 16
MARGIN = 0.2
MINDIST = 0.3


def randval(margin):
    return random.random() * (1.0 - 2 * margin) + margin


def generate_small_testset(input_filepath, output_filepath, negs_multi, margin, mindist):
    lines = fileproc.freadlines(input_filepath)
    equal_lines = []

    # Filter for equal lines only
    for l in lines:
        tokens = l.split(' ')
        grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens
        sim = int(sim)

        if sim:
            equal_lines.append(l)

    all_lines = list(equal_lines)

    # Generate random negative examples with random patches from the same image
    # TODO: Use different images....., have to modify the data layer?
    for l in progress_bar(equal_lines):
        tokens = l.split(' ')
        grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens

        for j in range(negs_multi):
            while True:
                vals = np.array([randval(margin) for x in range(4)])

                if np.linalg.norm(vals[:2] - vals[2:]) < mindist:
                    sim = '0'
                    p1x, p1y, p2x, p2y = [str(x) for x in vals]

                    all_lines.append(' '.join([grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y]))
                    break

    fileproc.fwritelines(output_filepath, all_lines)


if __name__ == "__main__":
    generate_small_testset(TRAINFILEPATH_PATCH, TRAINNEGSFILEPATH_PATCH, NEGS_MULTI, MARGIN, MINDIST)
    generate_small_testset(TRAINFILEPATH_PATCH_RGB, TRAINNEGSFILEPATH_PATCH_RGB, NEGS_MULTI, MARGIN, MINDIST)
