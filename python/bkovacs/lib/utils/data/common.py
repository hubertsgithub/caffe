import itertools
import math
import os

import numpy as np
import scipy as sp
from PIL import Image

import png

SCALE16BIT = 65535.


def print_array_info(array, array_name=''):
    if array_name != '':
        print 'ARRAY: {0}'.format(array_name)

    print 'Shape: {0}'.format(array.shape)
    print 'Min: {0}'.format(np.min(array))
    print 'Max: {0}'.format(np.max(array))
    if sp.sparse.issparse(array):
        print 'Avg: {0}'.format(array.mean())
    else:
        print 'Avg: {0}'.format(np.average(array))


def load_png(fname):
    reader = png.Reader(fname)
    w, h, pngdata, params = reader.read()
    image = np.vstack(itertools.imap(np.uint16, pngdata))
    if image.size == 3*w*h:
        image = np.reshape(image, (h, w, 3))
    print fname
    print_array_info(image)
    return image.astype(float) / SCALE16BIT


def save_png(array, fname):
    greyscale = (array.ndim == 2)
    array = (SCALE16BIT*array).astype(np.uint32)
    m, n = array.shape[:2]
    f = open(fname, 'wb')
    writer = png.Writer(n, m, greyscale=greyscale, bitdepth=16)
    if not greyscale:
        array = array.reshape(m, n*3)
    writer.write(f, array)


def load_image(filename, is_srgb=True):
    """ Load an image that is either linear or sRGB-encoded. """

    if not filename:
        raise ValueError('Empty filename')
    image = np.asarray(Image.open(filename)).astype(np.float) / 255.0
    if is_srgb:
        return srgb_to_rgb(image)
    else:
        return image


def save_image(filename, imagearr, is_srgb=True):
    """
    The image values should be in [0.0, 1.0]
    Save an image that is either linear or sRGB-encoded.
    """

    if not filename:
        raise ValueError('Empty filename')

    if is_srgb:
        imagearr = rgb_to_srgb(imagearr)

    if not (imagearr.ndim == 2 or (imagearr.ndim == 3 and (imagearr.shape[2] == 1 or imagearr.shape[2] == 3))):
        raise ValueError('Invalid image dimensions: {0}'.format(imagearr.shape))

    if imagearr.ndim == 3 and imagearr.shape[2] == 1:
        imagearr = np.squeeze(imagearr, axis=2)

    imagearr *= 255.0
    imagearr = np.asarray(imagearr, dtype=np.uint8)

    image = Image.fromarray(imagearr)
    image.save(filename)


def srgb_to_rgb(srgb):
    """ Convert an sRGB image to a linear RGB image """

    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def rgb_to_srgb(rgb):
    """ Convert an image from linear RGB to sRGB.

    :param rgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(rgb)
    idx0 = rgb <= 0.0031308
    idx1 = rgb > 0.0031308
    ret[idx0] = rgb[idx0] * 12.92
    ret[idx1] = np.power(1.055 * rgb[idx1], 1.0 / 2.4) - 0.055
    return ret


def compute_crop_tuple(width, height, cropw, croph, croplen):
    cropminus = math.floor(croplen / 2.)
    cropplus = math.ceil(croplen / 2.)
    fromw = int(max(0, cropw - cropminus))
    tow = int(min(width, cropw + cropplus))
    fromh = int(max(0, croph - cropminus))
    toh = int(min(height, croph + cropplus))
    tup = (fromw, fromh, tow, toh)

    return tup


def resize_and_crop_channel(ch_arr, resize, crop, keep_aspect_ratio=False, use_greater_side=True):
    '''
    Resizes and crops the middle of the provided image channel array
    '''
    if ch_arr.ndim != 2:
        raise ValueError('The provided image array should be two dimensional! Provided array dimensions: {0}'.format(ch_arr.shape))

    image = Image.fromarray(ch_arr)

    if resize is not None:
        if keep_aspect_ratio:
            w, h = image.size
            if (w > h and use_greater_side) or (w < h and not use_greater_side):
                r = float(resize) / w
                dim = (resize, int(h * r))
                image = image.resize(dim, Image.BILINEAR)
            else:
                r = float(resize) / h
                dim = (int(w * r), resize)
                image = image.resize(dim, Image.BILINEAR)
        else:
            image = image.resize((resize, resize), Image.BILINEAR)

    if crop is not None:
        w, h = image.size
        middle = [x / 2 for x in image.size]
        tup = compute_crop_tuple(w, h, middle[0], middle[1], crop)
        image = image.crop(tup)

    ret = np.array(image)
    return ret


def resize_and_crop_image(arr, resize, crop, keep_aspect_ratio=False, use_greater_side=True):
    if arr.ndim == 3:
        rets = []
        for c in range(3):
            rets.append(resize_and_crop_channel(arr[:, :, c], resize, crop, keep_aspect_ratio, use_greater_side))
        res_arr = np.dstack(rets)
    elif arr.ndim == 2:
        res_arr = resize_and_crop_channel(arr, resize, crop, keep_aspect_ratio, use_greater_side)
    else:
        raise ValueError('The provided image array should have either 1 or 3 channels!')

    return res_arr


def crop_image_channel(ch_arr, cropw, croph, croplen):
    if ch_arr.ndim != 2:
        raise ValueError('The provided image array should be two dimensional! Provided array dimensions: {0}'.format(ch_arr.shape))

    image = Image.fromarray(ch_arr)

    w, h = image.size
    tup = compute_crop_tuple(w, h, cropw, croph, croplen)
    image = image.crop(tup)

    ret = np.array(image)
    return ret


def crop_image(arr, cropw, croph, croplen):
    if arr.ndim == 3:
        rets = []
        for c in range(3):
            rets.append(crop_image_channel(arr[:, :, c], cropw, croph, croplen))
        res_arr = np.dstack(rets)
    elif arr.ndim == 2:
        res_arr = crop_image_channel(arr, cropw, croph, croplen)
    else:
        raise ValueError('The provided image array should have either 1 or 3 channels!')

    return res_arr


def compute_chromaticity_image(image):
    if image.ndim != 3:
        raise ValueError('The image should have 3 channels (RGB)!')

    sumimg = np.sum(image, axis=2)
    sumimg = np.clip(sumimg, 0.01, np.inf)
    chrom = image / sumimg[:, :, np.newaxis]

    return chrom


def compute_color_reflectance(gray_refl, img):
    chromimg = compute_chromaticity_image(img)

    # multiply by 3, because we don't do that when computing the chromaticity image
    return gray_refl[:, :, np.newaxis] * chromimg


def ensuredir(dirpath):
    if not os.path.exists(dirpath):
        os.makedirs(dirpath)

