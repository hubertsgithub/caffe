import random

from lib.utils.misc.pathresolver import acrp
from lib.utils.data import fileproc

TESTFILEPATH_PATCH = acrp('data/iiw-dataset/test.txt')
TESTSMALLFILEPATH_PATCH = acrp('data/iiw-dataset/test-small.txt')
# number for each class
SAMPLECOUNT_PATCH = 500
CLASSNUMBER_PATCH = 2

TESTFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/test-rgb.txt')
TESTSMALLFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/test-rgb-small.txt')
# number for each class
SAMPLECOUNT_PATCH_RGB = 500
CLASSNUMBER_PATCH_RGB = 2

TESTFILEPATH_BINSEARCH = acrp('data/iiw-dataset/test-bin-threshold-all.txt')
TESTSMALLFILEPATH_BINSEARCH = acrp('data/iiw-dataset/test-bin-threshold-all-small.txt')
# number for each class
SAMPLECOUNT_BINSEARCH = 250
CLASSNUMBER_BINSEARCH = 3


def generate_small_testset(input_filepath, output_filepath, classnum, extract_classid_func, samplecount):
    lines = fileproc.freadlines(input_filepath)
    lines_by_class = [[] for c in range(classnum)]

    for l in lines:
        tokens = l.split(' ')
        classid = extract_classid_func(tokens)
        lines_by_class[classid].append(l)

    samples = []

    for c in range(classnum):
        samples.append(random.sample(lines_by_class[c], samplecount))

    zipped = zip(*samples)
    sampled_lines = [e for t in zipped for e in t]

    fileproc.fwritelines(output_filepath, sampled_lines)


def extract_patches_data(tokens):
    grayimg_path, chromimg_path, classid, p1x, p1y, p2x, p2y = tokens
    return int(classid)


def extract_binsearch_data(tokens):
    shading_file, refl_file, class_idx = tokens
    return int(class_idx)


if __name__ == "__main__":
    generate_small_testset(TESTFILEPATH_PATCH, TESTSMALLFILEPATH_PATCH, CLASSNUMBER_PATCH, extract_patches_data, SAMPLECOUNT_PATCH)
    generate_small_testset(TESTFILEPATH_PATCH_RGB, TESTSMALLFILEPATH_PATCH_RGB, CLASSNUMBER_PATCH_RGB, extract_patches_data, SAMPLECOUNT_PATCH_RGB)
    generate_small_testset(TESTFILEPATH_BINSEARCH, TESTSMALLFILEPATH_BINSEARCH, CLASSNUMBER_BINSEARCH, extract_binsearch_data, SAMPLECOUNT_BINSEARCH)
