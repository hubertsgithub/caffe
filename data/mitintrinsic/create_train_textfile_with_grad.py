import os
from os import listdir
from os.path import exists

import cv2
import numpy as np

import poisson
import itertools
import png

scale = 65535.

def print_array_info(array):
    print 'Shape: {0}'.format(array.shape)
    print 'Min: {0}'.format(np.min(array))
    print 'Max: {0}'.format(np.max(array))
    print 'Avg: {0}'.format(np.average(array))

def load_png(fname):
    reader = png.Reader(fname)
    w, h, pngdata, params = reader.read()
    image = np.vstack(itertools.imap(np.uint16, pngdata))
    if image.size == 3*w*h:
        image = np.reshape(image, (h, w, 3))
    print fname
    print_array_info(image)
    return image.astype(float) / scale

def computegradimgs(shading, reflectance, mask):
    #mask = np.mean(maskimg, axis = 2)

    #images = map(lambda image: np.clip(image, .001, np.infty), images)
    # Convert to grayscale
    if len(shading.shape) == 3:
        shading = np.mean(shading, axis = 2)
    if len(reflectance.shape) == 3:
        reflectance = np.mean(reflectance, axis = 2)

    # Compute log images
    #images = map(lambda image: np.where(mask, np.log(image), 0.), images)
    # Compute gradients
    s_y, s_x = poisson.get_gradients(shading)
    r_y, r_x = poisson.get_gradients(reflectance)

    # Shading gradient -> 1, reflectance gradient -> 0
    epsilon = 0.01
    b_y = np.where(np.logical_or(np.abs(s_y) > np.abs(r_y), np.abs(r_y) < epsilon), 1., 0.)
    b_x = np.where(np.logical_or(np.abs(s_x) > np.abs(r_x), np.abs(r_x) < epsilon), 1., 0.)

    b_y = b_y * 255.0
    b_x = b_x * 255.0

    return b_x, b_y

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
    filepaths.append(os.path.join(origparentdirpath, 'shading.png'))
    filepaths.append(os.path.join(origparentdirpath, 'reflectance.png'))
    filepaths.append(os.path.join(origparentdirpath, 'mask.png'))

    images = [load_png(fp) for fp in filepaths]
    b_x, b_y = computegradimgs(images[0], images[1], images[2])

    cv2.imwrite(os.path.join(origparentdirpath, 'thresholdx.png'), b_x)
    cv2.imwrite(os.path.join(origparentdirpath, 'thresholdy.png'), b_y)

    filepaths = []
    filepaths.append(os.path.join(origparentdirpath, 'shading-converted.png'))
    filepaths.append(os.path.join(origparentdirpath, 'reflectance-converted.png'))
    filepaths.append(os.path.join(origparentdirpath, 'mask-converted.png'))

    images = [load_png(fp) for fp in filepaths]
    b_x, b_y = computegradimgs(images[0], images[1], images[2])

    convertedfilepathx = os.path.join(origparentdirpath, 'gradbinary-x-converted.png')
    convertedfilepathy = os.path.join(origparentdirpath, 'gradbinary-y-converted.png')
    cv2.imwrite(convertedfilepathx, b_x)
    cv2.imwrite(convertedfilepathy, b_y)

    f.write('{0} {1} {2} {3}\n'.format(os.path.join(origparentdirpath, 'diffuse-converted-gamma.png'),
                                          convertedfilepathx, convertedfilepathy,
                                          os.path.join(origparentdirpath, 'mask-converted.png')))

    if first:
        f = open('data/mitintrinsic/train_with_gradient.txt', 'w')
        first = False

f.close()

print "Done!"
