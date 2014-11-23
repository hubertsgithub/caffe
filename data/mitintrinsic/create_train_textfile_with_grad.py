import os
from os import listdir
from os.path import exists

import cv2
import numpy as np

import intrinsic
import poisson

origpath = 'data/mitintrinsic/data'

origdirnames = listdir(origpath)
origdirnames.sort()

f = open('data/mitintrinsic/val_with_gradient.txt', 'w')

first = True
for dir in origdirnames:
    origparentdirpath = os.path.join(origpath, dir)

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    filepaths = []
    filepaths.append(os.path.join(origparentdirpath, 'shading-converted.png'))
    filepaths.append(os.path.join(origparentdirpath, 'reflectance-converted.png'))

    images = [intrinsic.load_png(fp) for fp in filepaths]
    mask = intrinsic.load_png(os.path.join(origparentdirpath, 'mask-converted.png'))
    mask = np.mean(mask, axis = 2)

    images = map(lambda image: np.clip(image, .001, np.infty), images)
    # Convert to grayscale
    images = map(lambda image: np.mean(image, axis = 2), images)

    # Compute log images
    images = map(lambda image: np.where(mask, np.log(image), 0.), images)
    # Compute gradients
    s_y, s_x = poisson.get_gradients(images[0])
    r_y, r_x = poisson.get_gradients(images[1])

    # Shading gradient -> 1, reflectance gradient -> 0
    b_y = np.where(np.abs(s_y) > np.abs(r_y), 1., 0.)
    b_x = np.where(np.abs(s_x) > np.abs(r_x), 1., 0.)

    #b_y = b_y * 255.0
    #b_x = b_x * 255.0

    convertedfilepathx = os.path.join(origparentdirpath, 'gradbinary-x-converted.png')
    convertedfilepathy = os.path.join(origparentdirpath, 'gradbinary-y-converted.png')
    cv2.imwrite(convertedfilepathx, b_x)
    cv2.imwrite(convertedfilepathy, b_y)

    f.write('{0} {1} {2} {3}\n'.format(os.path.join(origparentdirpath, 'diffuse-converted.png'),
                                          convertedfilepathx, convertedfilepathy,
                                          os.path.join(origparentdirpath, 'mask-converted.png')))

    if first:
        f = open('data/mitintrinsic/train_with_gradient.txt', 'w')
        first = False

f.close()

print "Done!"
