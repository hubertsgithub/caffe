from PIL import Image
import png
import itertools
import numpy as np
import scipy as sp
import poisson

SCALE16BIT = 65535.

def print_array_info(array):
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

def resize_and_crop_channel(ch_arr, resize, crop, keep_aspect_ratio=False):
    '''
    Resizes and crops the middle of the provided image channel array
    '''
    if len(ch_arr.shape) != 2:
        raise ValueError('The provided image array should be two dimensional! Provided array dimensions: {0}'.format(ch_arr.shape))

    image = Image.fromarray(ch_arr)
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

def resize_and_crop_image(arr, resize, crop, keep_aspect_ratio=False):
    if len(arr.shape) == 3:
        rets = []
        for c in range(3):
            rets.append(resize_and_crop_channel(arr[:, :, c], resize, crop, keep_aspect_ratio))
        res_arr = np.dstack(rets)
    elif len(arr.shape) == 2:
        res_arr = resize_and_crop_channel(arr, resize, crop, keep_aspect_ratio)
    else:
        raise ValueError('The provided image array should have either 1 or 3 channels!')

    return res_arr

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

def compute_chromaticity_image(image):
    if len(image.shape) != 3:
        raise ValueError('The image should have 3 channels (RGB)!')

    sumimg = np.sum(image, axis = 2)
    sumimg = np.clip(sumimg, 0.01, np.inf)
    chrom = image / sumimg[:, :, np.newaxis]

    return chrom
