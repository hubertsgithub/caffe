from PIL import Image
import png
import itertools
import numpy as np
import poisson

SCALE16BIT = 65535.

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

def resize_and_crop_channel(ch_arr, resize, crop):
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
