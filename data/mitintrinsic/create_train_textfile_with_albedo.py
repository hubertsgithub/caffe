from os import listdir
from os.path import exists
import os
from PIL import Image
import png
import itertools
import numpy as np

scale = 65535.
origpath = 'data/mitintrinsic/data'
resize = 190
crop = 190
gamma = 2.2

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

def save_png(array, fname):
    greyscale = (array.ndim == 2)
    array = (scale*array).astype(np.uint32)
    m, n = array.shape[:2]
    f = open(fname, 'wb')
    writer = png.Writer(n, m, greyscale=greyscale, bitdepth=16)
    if not greyscale:
        array = array.reshape(m, n*3)
    writer.write(f, array)

def resize_and_crop_channel(ch_arr):
    image = Image.fromarray(ch_arr)
    keep_aspect_ratio = False
    if keep_aspect_ratio:
        w, h = image.size
        if w > h:
            r = float(resize) / w
            dim = (resize, int(h * r))
            image = image.resize(dim, Image.BILINEAR)
        else:
            r = float(resize) / h
            dim = (int(w * r), resize)
            image = image.resize(dim, Image.BILINEAR)
    else:
        image = image.resize((resize, resize), Image.BILINEAR)

    w, h = image.size
    middle = [x / 2 for x in image.size]
    fromw = max(0, middle[0] - crop / 2)
    tow = min(w, middle[0] + crop / 2)
    fromh = max(0, middle[1] - crop / 2)
    toh = min(h, middle[1] + crop / 2)
    tup = (fromw, fromh, tow, toh)
    print tup
    image = image.crop(tup)

    ret = np.array(image)
    return ret

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

        arr = load_png(filepath)
        if len(arr.shape) == 3:
            rets = []
            for c in range(3):
                rets.append(resize_and_crop_channel(arr[:, :, c]))
            res_arr = np.dstack(rets)
        else:
            res_arr = resize_and_crop_channel(arr)

        fileName, fileExtension = os.path.splitext(filepath)
        convertedfilepath = fileName + '-converted' + fileExtension

        # save the image and put the gamma corrected image into the convertedfilepaths
        if first_file:
            save_png(res_arr, convertedfilepath)
            res_arr = np.power(res_arr, 1./gamma)
            convertedfilepath = fileName + '-converted-gamma' + fileExtension
            first_file = False

        convertedfilepaths.append(convertedfilepath)
        save_png(res_arr, convertedfilepath)

    f.write('{0} {1} {2} {3} {4}\n'.format(convertedfilepaths[0], convertedfilepaths[0], convertedfilepaths[1], convertedfilepaths[2], convertedfilepaths[3]))

    if first:
        f = open('data/mitintrinsic/train_with_albedo.txt', 'w')
        first = False

f.close()

print "Done!"
