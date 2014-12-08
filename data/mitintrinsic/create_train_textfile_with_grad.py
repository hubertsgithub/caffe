import os
from os import listdir
from os.path import exists

import cv2
import numpy as np

import common
import poisson

origpath = 'data/mitintrinsic/data'

origdirnames = listdir(origpath)
origdirnames.sort()

VALSET = ['box', 'paper1']

f_train = open('data/mitintrinsic/train_with_gradient.txt', 'w')
f_val = open('data/mitintrinsic/val_with_gradient.txt', 'w')

for dir in origdirnames:
    origparentdirpath = os.path.join(origpath, dir)

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    filepaths = []
    filepaths.append(os.path.join(origparentdirpath, 'shading.png'))
    filepaths.append(os.path.join(origparentdirpath, 'reflectance.png'))
    filepaths.append(os.path.join(origparentdirpath, 'mask.png'))

    images = [common.load_png(fp) for fp in filepaths]
    b_x, b_y = common.computegradimgs(images[0], images[1], images[2])

    cv2.imwrite(os.path.join(origparentdirpath, 'thresholdx.png'), b_x)
    cv2.imwrite(os.path.join(origparentdirpath, 'thresholdy.png'), b_y)

    filepaths = []
    filepaths.append(os.path.join(origparentdirpath, 'shading-converted.png'))
    filepaths.append(os.path.join(origparentdirpath, 'reflectance-converted.png'))
    filepaths.append(os.path.join(origparentdirpath, 'mask-converted.png'))

    images = [common.load_png(fp) for fp in filepaths]
    b_x, b_y = common.computegradimgs(images[0], images[1], images[2])

    convertedfilepathx = os.path.join(origparentdirpath, 'gradbinary-x-converted.png')
    convertedfilepathy = os.path.join(origparentdirpath, 'gradbinary-y-converted.png')
    cv2.imwrite(convertedfilepathx, b_x)
    cv2.imwrite(convertedfilepathy, b_y)

    if dir in VALSET:
        f = f_val
    else:
        f = f_train

    f.write('{0} {1} {2} {3} {4}\n'.format(os.path.join(origparentdirpath, 'diffuse-converted-gamma.png'),
                                        os.path.join(origparentdirpath, 'diffuse-converted-chrom.png'),
                                          convertedfilepathx, convertedfilepathy,
                                          os.path.join(origparentdirpath, 'mask-converted.png')))

f_train.close()
f_val.close()

print "Done!"
