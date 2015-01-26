import random

from lib.utils.misc.pathresolver import acrp
from lib.utils.data import fileproc

TRAINFILEPATH_PATCH = acrp('data/iiw-dataset/train.txt')
TESTFILEPATH_PATCH = acrp('data/iiw-dataset/test.txt')
TRAINFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/train-rgb.txt')
TESTFILEPATH_PATCH_RGB = acrp('data/iiw-dataset/test-rgb.txt')


def convert_txtfile_to_rgb(input_filepath, output_filepath):
    lines = fileproc.freadlines(input_filepath)
    converted_lines = []

    for l in lines:
        tokens = l.split(' ')
        grayimg_path, chromimg_path, sim, p1x, p1y, p2x, p2y = tokens
        orig_path = grayimg_path.replace('-gray', '')

        converted_lines.append(' '.join([orig_path, orig_path, sim, p1x, p1y, p2x, p2y]))

    fileproc.fwritelines(output_filepath, converted_lines)


if __name__ == "__main__":
    convert_txtfile_to_rgb(TRAINFILEPATH_PATCH, TRAINFILEPATH_PATCH_RGB)
    convert_txtfile_to_rgb(TESTFILEPATH_PATCH, TESTFILEPATH_PATCH_RGB)
