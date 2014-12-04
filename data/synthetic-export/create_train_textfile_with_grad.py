import os
from os import listdir
from os.path import exists

import cv2
import numpy as np

import poisson
import multilayer_exr

def computegradimgs(shading, reflectance, mask):
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

rootpath = 'data/synthetic-export'
datapath = os.path.join(rootpath, 'data')

datafilenames = listdir(datapath)
datafilenames.sort()

f = open(os.path.join(rootpath, 'val_with_gradient.txt'), 'w')

first = True
for filename in datafilenames:
    filepath = os.path.join(datapath, filename)

    if not exists(filepath):
        print filepath + " doesn't exist, moving on..."
        continue

    name, ext = os.path.splitext(filename)

    if not ext == '.exr':
        continue
    try:
        layers = multilayer_exr.open_multilayer_exr(filepath)
    except:
        print 'Something went wrong with image {0}'.format(filepath)
        continue

    shading = layers['diff_dir'] + layers['diff_ind']
    reflectance = layers['diff_col']
    mask = np.ones_like(shading)
    combined = shading * reflectance

    b_x, b_y = computegradimgs(shading, reflectance, mask)

    convertedfilepathx = os.path.join(datapath, name + '-gradbinary-x-converted.png')
    convertedfilepathy = os.path.join(datapath, name + '-gradbinary-y-converted.png')
    shadingpath = os.path.join(datapath, name + '-shading.png')
    reflectancepath = os.path.join(datapath, name + '-reflectance.png')
    maskpath = os.path.join(datapath, name + '-mask.png')
    combinedpath = os.path.join(datapath, name + '-combined.png')

    cv2.imwrite(convertedfilepathx, b_x)
    cv2.imwrite(convertedfilepathy, b_y)
    cv2.imwrite(shadingpath, multilayer_exr.rgb_to_srgb(shading))
    cv2.imwrite(reflectancepath, multilayer_exr.rgb_to_srgb(reflectance))
    cv2.imwrite(maskpath, multilayer_exr.srgb_to_rgb(mask))
    cv2.imwrite(combinedpath, multilayer_exr.srgb_to_rgb(combined))

    f.write('{0} {1} {2} {3}\n'.format(combinedpath,
                                          convertedfilepathx, convertedfilepathy,
                                          maskpath))

    if first:
        f = open(os.path.join(datapath, 'train_with_gradient.txt'), 'w')
        first = False

f.close()

print "Done!"
