import itertools
import os
import sys

import numpy as np
import png
import scipy as sp
from PIL import Image

sys.path.append('experiments')
sys.path.append('data')
import globals
import common
import poisson
import runcnn
import zhao2012

LOADROOTDIRMIT = 'data/mitintrinsic/'
LOADROOTDIRINDOOR = 'data/synthetic-export/'

############################### Data ###########################################


def load_png(fname):
    reader = png.Reader(fname)
    w, h, pngdata, params = reader.read()
    image = np.vstack(itertools.imap(np.uint16, pngdata))
    if image.size == 3*w*h:
        image = np.reshape(image, (h, w, 3))

    print fname
    common.print_array_info(image)

    image = image.astype(float)
    if globals.CHOOSEMIT:
        image = image / 255.

    return image


def load_object_helper(tag, condition):
    """Load an image of a given object as a NumPy array. The values condition may take are:

            'mask', 'original', 'diffuse', 'shading', 'reflectance', 'specular'

    'shading' returns a grayscale image, and all the other options return color images."""
    assert condition in ['mask', 'original', 'diffuse', 'shading', 'reflectance', 'specular', 'thresholdx', 'thresholdy']

    if globals.CHOOSEMIT:
        obj_dir = os.path.join(LOADROOTDIRMIT, 'data', tag)
        convert_str = '-converted'

        if condition == 'mask':
            filename = os.path.join(obj_dir, 'mask{0}.png'.format(convert_str))
            mask = load_png(filename)
            return (mask > 0)
        if condition == 'original':
            filename = os.path.join(obj_dir, 'original{0}.png'.format(convert_str))
            return load_png(filename)
        if condition == 'diffuse':
            filename = os.path.join(obj_dir, 'diffuse{0}.png'.format(convert_str))
            return load_png(filename)
        if condition == 'shading':
            filename = os.path.join(obj_dir, 'shading{0}.png'.format(convert_str))
            return load_png(filename)
        if condition == 'reflectance':
            filename = os.path.join(obj_dir, 'reflectance{0}.png'.format(convert_str))
            return load_png(filename)
        if condition == 'specular':
            filename = os.path.join(obj_dir, 'specular{0}.png'.format(convert_str))
            return load_png(filename)
        if condition == 'thresholdx':
            filename = os.path.join(obj_dir, 'gradbinary-x-converted.png'.format(convert_str))
            return load_png(filename)
        if condition == 'thresholdy':
            filename = os.path.join(obj_dir, 'gradbinary-y-converted.png'.format(convert_str))
            return load_png(filename)
    else:
        obj_dir = os.path.join(LOADROOTDIRINDOOR, 'data')
        if condition == 'mask':
            filename = os.path.join(obj_dir, '{0}-mask.png'.format(tag))
            mask = load_png(filename)
            return (mask > 0)
        if condition == 'original':
            raise ValueError('Unsupported value for indoors dataset')
        if condition == 'diffuse':
            filename = os.path.join(obj_dir, '{0}-combined.png'.format(tag))
            return load_png(filename)
        if condition == 'shading':
            filename = os.path.join(obj_dir, '{0}-shading.png'.format(tag))
            return np.mean(load_png(filename), axis=2)
        if condition == 'reflectance':
            filename = os.path.join(obj_dir, '{0}-reflectance.png'.format(tag))
            return load_png(filename)
        if condition == 'specular':
            raise ValueError('Unsupported value for indoors dataset')
        if condition == 'thresholdx':
            filename = os.path.join(obj_dir, '{0}-gradbinary-x-converted.png'.format(tag))
            return load_png(filename)
        if condition == 'thresholdy':
            filename = os.path.join(obj_dir, '{0}-gradbinary-y-converted.png'.format(tag))
            return load_png(filename)

# cache for efficiency because PyPNG is pure Python
cache = {}
def load_object(tag, condition):
    if (tag, condition) not in cache:
        cache[tag, condition] = load_object_helper(tag, condition)
    return cache[tag, condition]


def load_multiple(tag):
    """Load the images of a given object for all lighting conditions. Returns an
    m x n x 3 x 10 NumPy array, where the third dimension is the color channel and
    the fourth dimension is the image number."""
    assert CHOOSEMIT

    obj_dir = os.path.join(LOADROOTDIRMIT, 'data', tag)
    filename = os.path.join(obj_dir, 'light01.png')
    img0 = load_png(filename)
    result = np.zeros(img0.shape + (10,))

    for i in range(10):
        filename = os.path.join(obj_dir, 'light%02d.png' % (i+1))
        result[:,:,:,i] = load_png(filename)

    return result



############################# Error metric #####################################


def ssq_error(correct, estimate, mask):
    """Compute the sum-squared-error for an image, where the estimate is
    multiplied by a scalar which minimizes the error. Sums over all pixels
    where mask is True. If the inputs are color, each color channel can be
    rescaled independently."""
    assert correct.ndim == 2
    if np.sum(estimate**2 * mask) > 1e-5:
        alpha = np.sum(correct * estimate * mask) / np.sum(estimate**2 * mask)
    else:
        alpha = 0.
    return np.sum(mask * (correct - alpha*estimate) ** 2)

def local_error(correct, estimate, mask, window_size, window_shift):
    """Returns the sum of the local sum-squared-errors, where the estimate may
    be rescaled within each local region to minimize the error. The windows are
    window_size x window_size, and they are spaced by window_shift."""
    M, N = correct.shape[:2]
    ssq = total = 0.
    for i in range(0, M - window_size + 1, window_shift):
        for j in range(0, N - window_size + 1, window_shift):
            correct_curr = correct[i:i+window_size, j:j+window_size]
            estimate_curr = estimate[i:i+window_size, j:j+window_size]
            mask_curr = mask[i:i+window_size, j:j+window_size]
            ssq += ssq_error(correct_curr, estimate_curr, mask_curr)
            total += np.sum(mask_curr * correct_curr**2)
    assert -np.isnan(ssq/total)

    return ssq / total

def score_image(true_shading, true_refl, estimate_shading, estimate_refl, mask, window_size=20):
    return 0.5 * local_error(true_shading, estimate_shading, mask, window_size, window_size//2) + \
           0.5 * local_error(true_refl, estimate_refl, mask, window_size, window_size//2)


################################## Algorithms ##################################

def retinex(image, mask, threshold, L1=False):
    image = np.clip(image, 3., np.infty)
    log_image = np.where(mask, np.log(image), 0.)
    i_y, i_x = poisson.get_gradients(log_image)

    r_y = np.where(np.abs(i_y) > threshold, i_y, 0.)
    r_x = np.where(np.abs(i_x) > threshold, i_x, 0.)

    if L1:
        log_refl = poisson.solve_L1(r_y, r_x, mask)
    else:
        log_refl = poisson.solve(r_y, r_x, mask)
    refl = mask * np.exp(log_refl)

    return np.where(mask, image / refl, 0.), refl

def retinex_with_thresholdimage(image, mask, threshold_image_x, threshold_image_y, L1=False):
    image = np.mean(image, axis=2)
    image = np.clip(image, 3., np.infty)
    log_image = np.where(mask, np.log(image), 0.)
    i_y, i_x = poisson.get_gradients(log_image)

    r_y = (1. - threshold_image_y) * i_y
    r_x = (1. - threshold_image_x) * i_x
    #r_y = np.where(threshold_image_y < 0.5, i_y, 0.)
    #r_x = np.where(threshold_image_x < 0.5, i_x, 0.)

    if L1:
        log_refl = poisson.solve_L1(r_y, r_x, mask)
    else:
        log_refl = poisson.solve(r_y, r_x, mask)

    common.print_array_info(log_refl)
    #log_refl = np.clip(log_refl, -40., 20.)
    refl = mask * np.exp(log_refl)

    return np.where(mask, image / refl, 0.), refl

def project_gray(i_y):
    i_y_mean = np.mean(i_y, axis=2)
    result = np.zeros(i_y.shape)
    for i in range(3):
        result[:,:,i] = i_y_mean
    return result

def project_chromaticity(i_y):
    return i_y - project_gray(i_y)

def color_retinex(image, mask, threshold_gray, threshold_color, L1=False):
    image = np.clip(image, 3., np.infty)

    log_image = np.log(image)
    i_y_orig, i_x_orig = poisson.get_gradients(log_image)
    i_y_gray, i_y_color = project_gray(i_y_orig), project_chromaticity(i_y_orig)
    i_x_gray, i_x_color = project_gray(i_x_orig), project_chromaticity(i_x_orig)

    image_grayscale = np.mean(image, axis=2)
    image_grayscale = np.clip(image_grayscale, 3., np.infty)
    log_image_grayscale = np.log(image_grayscale)
    i_y, i_x = poisson.get_gradients(log_image_grayscale)

    norm = np.sqrt(np.sum(i_y_color**2, axis=2))
    i_y_match = (norm > threshold_color) + (np.abs(i_y_gray[:,:,0]) > threshold_gray)

    norm = np.sqrt(np.sum(i_x_color**2, axis=2))
    i_x_match = (norm > threshold_color) + (np.abs(i_x_gray[:,:,0]) > threshold_gray)

    r_y = np.where(i_y_match, i_y, 0.)
    r_x = np.where(i_x_match, i_x, 0.)

    if L1:
        log_refl = poisson.solve_L1(r_y, r_x, mask)
    else:
        log_refl = poisson.solve(r_y, r_x, mask)
    refl =  np.exp(log_refl)

    return image_grayscale / refl, refl

def weiss(image, multi_images, mask, L1=False):
    multi_images = np.clip(multi_images, 3., np.infty)
    log_multi_images = np.log(multi_images)

    i_y_all, i_x_all = poisson.get_gradients(log_multi_images)
    r_y = np.median(i_y_all, axis=2)
    r_x = np.median(i_x_all, axis=2)
    if L1:
        log_refl = poisson.solve_L1(r_y, r_x, mask)
    else:
        log_refl = poisson.solve(r_y, r_x, mask)
    refl = np.where(mask, np.exp(log_refl), 0.)
    shading = np.where(mask, image / refl, 0.)

    return shading, refl

def weiss_retinex(image, multi_images, mask, threshold, L1=False):
    multi_images = np.clip(multi_images, 3., np.infty)
    log_multi_images = np.log(multi_images)

    i_y_all, i_x_all = poisson.get_gradients(log_multi_images)
    r_y = np.median(i_y_all, axis=2)
    r_x = np.median(i_x_all, axis=2)

    r_y *= (np.abs(r_y) > threshold)
    r_x *= (np.abs(r_x) > threshold)
    if L1:
        log_refl = poisson.solve_L1(r_y, r_x, mask)
    else:
        log_refl = poisson.solve(r_y, r_x, mask)
    refl = np.where(mask, np.exp(log_refl), 0.)
    shading = np.where(mask, image / refl, 0.)

    return shading, refl


def zhao2012algo(image, mask, threshold, L1=False):
    image = image / np.max(image)
    image = np.where(mask[:, :, np.newaxis], image ** (1./2.2), 1.)
    image = image * 255.0
    pil_image = Image.fromarray(image.astype(np.uint8))
    texture_patch_distance = 0.0003
    texture_patch_variance = 0.03
    gamma = False
    r, s = zhao2012.run(pil_image, threshold, texture_patch_distance, texture_patch_variance, gamma)
    reflscaled = np.array(r) / 255.0
    refl = np.mean(reflscaled, axis=2) * mask
    shadingscaled = np.array(s) / 255.0
    shading = np.mean(shadingscaled, axis=2) * mask
    print 'min: {0}, max: {1}, mean: {2}'.format(np.min(reflscaled), np.max(reflscaled), np.mean(reflscaled))
    print 'min: {0}, max: {1}, mean: {2}'.format(np.min(shadingscaled), np.max(shadingscaled), np.mean(shadingscaled))

    return shading, refl


#################### Wrapper classes for experiments ###########################


class BaselineEstimator:
    """Assume every image is entirely shading or entirely reflectance."""
    def __init__(self, mode, L1=False):
        assert mode in ['refl', 'shading']
        self.mode = mode

    def estimate_shading_refl(self, image, mask, L1=False):
        if self.mode == 'refl':
            refl = image
            shading = 1. * mask
        else:
            refl = 1. * mask
            shading = image
        return shading, refl

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        image = np.mean(image, axis=2)
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{'mode': m} for m in ['shading', 'refl']]


class GrayscaleRetinexEstimator:
    def __init__(self, threshold):
        self.threshold = threshold

    def estimate_shading_refl(self, image, mask, L1=False):
        return retinex(image, mask, self.threshold, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        image = np.mean(image, axis=2)
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{'threshold': t} for t in np.logspace(-3., 1., 15)]

def run_retinex_with_chrom_cnn_model(image, mask, model_name, model_iteration, L1):
    MODELPATH = 'ownmodels/mitintrinsic'
    MODEL_FILE = os.path.join(MODELPATH, 'deploy_{0}.prototxt'.format(model_name))
    PRETRAINED = os.path.join(MODELPATH, 'snapshots', 'caffenet_train_{0}_iter_{1}.caffemodel'.format(model_name, model_iteration))

    gamma_corrected_image = image / 255. #np.power(image / 255., 1/2.2)
    chrom_image = common.compute_chromaticity_image(gamma_corrected_image)
    gamma_corrected_image = np.mean(gamma_corrected_image, axis=2)
    threshold_image_x, threshold_image_y = runcnn.predict_thresholds(MODEL_FILE, PRETRAINED, [gamma_corrected_image, chrom_image])

    return retinex_with_thresholdimage(image, mask, threshold_image_x, threshold_image_y, L1)

class GrayscaleRetinexWithThresholdImageChromSmallNetEstimator:
    def estimate_shading_refl(self, image, mask, L1=False):
        return run_retinex_with_chrom_cnn_model(image, mask, 'gradient_pad_chrom', '50000', L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]

class GrayscaleRetinexWithThresholdImageChromBigNetEstimator:
    def estimate_shading_refl(self, image, mask, L1=False):
        return run_retinex_with_chrom_cnn_model(image, mask, 'gradient_pad_chrom2', '50000', L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]

class GrayscaleRetinexWithThresholdImageChromBigNetConcatEstimator:
    def estimate_shading_refl(self, image, mask, L1=False):
        return run_retinex_with_chrom_cnn_model(image, mask, 'gradient_pad_chrom_concat', '50000', L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]

class GrayscaleRetinexWithThresholdImageChromBigNetConcatMaxpoolEstimator:
    def estimate_shading_refl(self, image, mask, L1=False):
        return run_retinex_with_chrom_cnn_model(image, mask, 'gradient_pad_chrom_concat_maxpool', '50000', L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]

class GrayscaleRetinexWithThresholdImageRGBEstimator:
    def estimate_shading_refl(self, image, mask, L1=False):
        MODELPATH = 'ownmodels/mitintrinsic'
        MODEL_FILE = os.path.join(MODELPATH, 'deploy_gradient_pad.prototxt')
        PRETRAINED = os.path.join(MODELPATH, 'snapshots', 'caffenet_train_gradient_pad_iter_130000.caffemodel')

        gamma_corrected_image = image / 255. #np.power(image / 255., 1/2.2)
        threshold_image_x, threshold_image_y = runcnn.predict_thresholds(MODEL_FILE, PRETRAINED, [gamma_corrected_image])

        return retinex_with_thresholdimage(image, mask, threshold_image_x, threshold_image_y, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]

class GrayscaleRetinexWithThresholdImageGroundTruthEstimator:
    def estimate_shading_refl(self, image, mask, threshold_image_x, threshold_image_y, L1=False):
        return retinex_with_thresholdimage(image, mask, threshold_image_x, threshold_image_y, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        threshold_image_x = load_object(tag, 'thresholdx')
        threshold_image_y = load_object(tag, 'thresholdy')
        return image, mask, threshold_image_x, threshold_image_y

    @staticmethod
    def param_choices():
        return [{}]

class ColorRetinexEstimator:
    def __init__(self, threshold_gray, threshold_color, L1=False):
        self.threshold_gray = threshold_gray
        self.threshold_color = threshold_color

    def estimate_shading_refl(self, image, mask, L1=False):
        return color_retinex(image, mask, self.threshold_gray, self.threshold_color, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{'threshold_gray': tg, 'threshold_color': tc}
                for tg in np.logspace(-1.5, 0., 5)
                for tc in np.logspace(-1.5, 0., 5)]


class WeissEstimator:
    def estimate_shading_refl(self, image, multi_images, mask, L1=False):
        return weiss(image, multi_images, mask, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        image = np.mean(image, axis=2)
        mask = load_object(tag, 'mask')
        multi_images = load_multiple(tag)
        multi_images = np.mean(multi_images, axis=2)
        return image, multi_images, mask

    @staticmethod
    def param_choices():
        return [{}]


class WeissRetinexEstimator:
    def __init__(self, threshold=0.1, L1=False):
        self.threshold = threshold

    def estimate_shading_refl(self, image, multi_images, mask, L1=False):
        return weiss_retinex(image, multi_images, mask, self.threshold, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        image = np.mean(image, axis=2)
        mask = load_object(tag, 'mask')
        multi_images = load_multiple(tag)
        multi_images = np.mean(multi_images, axis=2)
        return image, multi_images, mask

    @staticmethod
    def param_choices():
        return [{'threshold': t} for t in np.logspace(-3., 1., 15)]


class Zhao2012Estimator:
    def __init__(self, threshold=0.001, L1=False):
        self.threshold = threshold

    def estimate_shading_refl(self, image, mask, L1=False):
        return zhao2012algo(image, mask, self.threshold, L1)

    @staticmethod
    def get_input(tag):
        image = load_object(tag, 'diffuse')
        mask = load_object(tag, 'mask')
        return image, mask

    @staticmethod
    def param_choices():
        return [{}]#'threshold': t} for t in np.logspace(-3., 1., 15)]
