import os
from os import listdir
from os.path import exists

import scipy as sp
import numpy as np

from lib.utils.data import common
from lib.utils.data import multilayer_exr
from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.progressbaraux import progress_bar

resize = 800


def scale_then_to_srgb(image):
    image = image / np.percentile(image, 99.9)
    image = np.clip(image, 0.0, 1.0)
    image = image ** (1 / 2.2)
    return 255.0 * image


if __name__ == '__main__':
    rootpath = acrp('data/synthetic-export')
    datapath = os.path.join(rootpath, 'data')
    datapathtxt = 'data/synthetic-export/data'
    CONVERTIMAGES = False

    datafilenames = listdir(datapath)
    datafilenames.sort()

    f = open(os.path.join(rootpath, 'test_with_gradient.txt'), 'w')

    first = True
    for filename in progress_bar(datafilenames):
        filepath = os.path.join(datapath, filename)

        if not exists(filepath):
            print filepath + " doesn't exist, moving on..."
            continue

        name, ext = os.path.splitext(filename)

        if not ext == '.exr':
            continue

        gradxpath = os.path.join(datapathtxt, name + '-gradbinary-x-converted.png')
        gradypath = os.path.join(datapathtxt, name + '-gradbinary-y-converted.png')
        shadingpath = os.path.join(datapathtxt, name + '-shading.png')
        reflectancepath = os.path.join(datapathtxt, name + '-reflectance.png')
        maskpath = os.path.join(datapathtxt, name + '-mask.png')
        combinedpath = os.path.join(datapathtxt, name + '-combined.png')
        graycombinedpath = os.path.join(datapathtxt, name + '-gray.png')
        chrompath = os.path.join(datapathtxt, name + '-chrom.png')

        if CONVERTIMAGES:
            layers = multilayer_exr.open_multilayer_exr(filepath)

            shading = layers['diff_dir'] + layers['diff_ind']
            reflectance = layers['diff_col']
            combined = shading * reflectance

            shading = common.resize_and_crop_image(shading, resize, crop=None, keep_aspect_ratio=True)
            reflectance = common.resize_and_crop_image(reflectance, resize, crop=None, keep_aspect_ratio=True)
            combined = common.resize_and_crop_image(combined, resize, crop=None, keep_aspect_ratio=True)

            chrom = common.compute_chromaticity_image(combined)

            gray_combined = np.mean(combined, axis=2)
            p = np.percentile(gray_combined, 0.01)
            mask = (gray_combined > p).astype(np.float32)

            b_x, b_y = common.computegradimgs(np.mean(shading, axis=2), np.mean(reflectance, axis=2), mask)

            sp.misc.imsave(acrp(gradxpath), b_x)
            sp.misc.imsave(acrp(gradypath), b_y)
            sp.misc.imsave(acrp(shadingpath), scale_then_to_srgb(shading))
            sp.misc.imsave(acrp(reflectancepath), scale_then_to_srgb(reflectance))
            sp.misc.imsave(acrp(maskpath), scale_then_to_srgb(mask))
            sp.misc.imsave(acrp(combinedpath), scale_then_to_srgb(combined))
            sp.misc.imsave(acrp(graycombinedpath), scale_then_to_srgb(gray_combined))
            sp.misc.imsave(acrp(chrompath), scale_then_to_srgb(chrom))

        f.write('{0} {1} {2} {3} {4}\n'.format(graycombinedpath, chrompath, gradxpath, gradypath, maskpath))

        if first:
            f = open(os.path.join(rootpath, 'train_with_gradient.txt'), 'w')
            first = False

    f.close()

    print "Done!"
