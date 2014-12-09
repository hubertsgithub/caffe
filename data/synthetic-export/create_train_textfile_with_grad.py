import os
import sys
from os import listdir
from os.path import exists

import scipy as sp
import numpy as np

sys.path.append('data')
import common
import multilayer_exr

resize_to_percent = 50
resize = 190
newsize = (resize, resize)

def scale_then_to_srgb(image):
    image = image / np.percentile(image, 99.9)
    image = np.clip(image, 0.0, 1.0)
    image = multilayer_exr.rgb_to_srgb(image)
    return 255.0 * image

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

    layers = multilayer_exr.open_multilayer_exr(filepath)

    shading = layers['diff_dir'] + layers['diff_ind']
    reflectance = layers['diff_col']
    combined = shading * reflectance

    shading = sp.misc.imresize(shading, newsize)
    reflectance = sp.misc.imresize(reflectance, newsize)
    combined = sp.misc.imresize(combined, newsize)

    gray_combined = np.mean(combined, axis=2)
    p = np.percentile(gray_combined, 0.01)
    mask = (gray_combined > p).astype(np.float32)

    b_x, b_y = common.computegradimgs(np.mean(shading, axis=2), np.mean(reflectance, axis=2), mask)

    convertedfilepathx = os.path.join(datapath, name + '-gradbinary-x-converted.png')
    convertedfilepathy = os.path.join(datapath, name + '-gradbinary-y-converted.png')
    shadingpath = os.path.join(datapath, name + '-shading.png')
    reflectancepath = os.path.join(datapath, name + '-reflectance.png')
    maskpath = os.path.join(datapath, name + '-mask.png')
    combinedpath = os.path.join(datapath, name + '-combined.png')

    print 'shading max: {0} min: {1}'.format(np.max(shading), np.min(shading))
    print 'reflectance max: {0} min: {1}'.format(np.max(reflectance), np.min(reflectance))
    print 'combined max: {0} min: {1}'.format(np.max(combined), np.min(combined))

    sp.misc.imsave(convertedfilepathx, b_x)
    sp.misc.imsave(convertedfilepathy, b_y)
    sp.misc.imsave(shadingpath, scale_then_to_srgb(shading))
    sp.misc.imsave(reflectancepath, scale_then_to_srgb(reflectance))
    sp.misc.imsave(maskpath, scale_then_to_srgb(mask))
    sp.misc.imsave(combinedpath, scale_then_to_srgb(combined))

    f.write('{0} {1} {2} {3}\n'.format(combinedpath, convertedfilepathx, convertedfilepathy, maskpath))

    if first:
        f = open(os.path.join(rootpath, 'train_with_gradient.txt'), 'w')
        first = False

f.close()

print "Done!"
