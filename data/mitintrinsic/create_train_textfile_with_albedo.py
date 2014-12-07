from os import listdir
from os.path import exists
import os
from PIL import Image
import png
import itertools
import numpy as np

import common

origpath = 'data/mitintrinsic/data'
resize = 190
crop = 190
gamma = 2.2

origdirnames = listdir(origpath)
origdirnames.sort()

f = open('data/mitintrinsic/val_with_albedo.txt', 'w')

first = True
for dir in origdirnames:
    origparentdirpath = origpath + '/' + dir

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    filepaths = []
    filepaths.append(origparentdirpath + '/' + 'diffuse.png')
    filepaths.append(origparentdirpath  + '/' + 'shading.png')
    filepaths.append(origparentdirpath + '/' + 'reflectance.png')
    filepaths.append(origparentdirpath + '/' + 'mask.png')

    first_file = True
    convertedfilepaths = []
    for filepath in filepaths:
        if not exists(filepath):
            print filepath + " doesn't exist, moving on..."
            continue

        arr = common.load_png(filepath)
        if len(arr.shape) == 3:
            rets = []
            for c in range(3):
                rets.append(common.resize_and_crop_channel(arr[:, :, c], resize, crop))
            res_arr = np.dstack(rets)
        else:
            res_arr = common.resize_and_crop_channel(arr, resize, crop)

        fileName, fileExtension = os.path.splitext(filepath)
        convertedfilepath = fileName + '-converted' + fileExtension

        # save the image and put the gamma corrected image into the convertedfilepaths
        if first_file:
            common.save_png(res_arr, convertedfilepath)
            res_arr = np.power(res_arr, 1./gamma)
            convertedfilepath = fileName + '-converted-gamma' + fileExtension
            first_file = False

        convertedfilepaths.append(convertedfilepath)
        common.save_png(res_arr, convertedfilepath)

    f.write('{0} {1} {2} {3} {4}\n'.format(convertedfilepaths[0], convertedfilepaths[0], convertedfilepaths[1], convertedfilepaths[2], convertedfilepaths[3]))

    if first:
        f = open('data/mitintrinsic/train_with_albedo.txt', 'w')
        first = False

f.close()

print "Done!"
