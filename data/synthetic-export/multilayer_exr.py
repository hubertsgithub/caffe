import Imath
import numpy as np
import OpenEXR
from PIL import Image

# The 14 layers we expect to find in a Multilayer OpenEXR file.
LAYER_CHANNELS = {
    'depth': ['RenderLayer.Depth.%s' % c for c in 'ZZZ'],
    'normal': ['RenderLayer.Normal.%s' % c for c in 'XYZ'],
}
for a in ['Combined', 'Emit', 'Env']:
    LAYER_CHANNELS[a.lower()] = ['RenderLayer.%s.%s' % (a, c) for c in 'RGB']
for a in ['Gloss', 'Diff', 'Trans']:
    for b in ['Dir', 'Ind', 'Col']:
        LAYER_CHANNELS['%s_%s' % (a.lower(), b.lower())] = [
            'RenderLayer.%s%s.%s' % (a, b, c) for c in 'RGB'
        ]


def open_multilayer_exr(filename, tonemap=False):
    """
    Load a multilayer OpenEXR file and return a dictionary mapping layers to
    either numpy float32 arrays (if ``tonemap=False``) or to 8bit PIL images
    (if ``tonemap=True``).

    :param filename: string filename
    :param tonemap: if ``True``, map to sRGB.  For the depth layer, normalize
        it to have max value of 1.  For all other layers, do not normalize.
    """

    print "Reading %s: %s layers..." % (filename, len(LAYER_CHANNELS))

    # Open the input file
    f = OpenEXR.InputFile(filename)
    header = f.header()

    # Compute the size
    dw = header['dataWindow']
    cols, rows = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    multilayer = {}

    # load channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    for key, channels in LAYER_CHANNELS.iteritems():
        print "Loading layer %s..." % key
        image = np.empty((rows, cols, 3), dtype=np.float32)
        for (i, c) in enumerate(channels):
            data = f.channel(c, FLOAT)
            image[:, :, i] = np.fromstring(data, dtype=np.float32) \
                .reshape((rows, cols))
        multilayer[key] = image

    # resize and tonemap
    if tonemap:
        for key, channels in LAYER_CHANNELS.iteritems():
            print "Tonemapping layer %s..." % key
            image = multilayer[key]
            if key == "depth":
                image /= np.max(image[np.isfinite(image)])

            # convert to sRGB PIL
            image = numpy_to_pil(rgb_to_srgb(np.clip(image, 0.0, 1.0)))
            multilayer[key] = image

    return multilayer


def open_multilayer_exr_layers(inputfile, layers):
    """
    Load a list of images, each corresponding to a layer of an OpenEXR file.
    This is more efficient than loading all layers (in ``open_multilayer_exr``).

    Note that "layer" does not correspond to a single color channel, like "R",
    but rather, a group of 3 color channels.

    :param inputfile: string filename
    :param layers: list of string layer names
    """
    f = OpenEXR.InputFile(inputfile)
    header = f.header()
    dw = header['dataWindow']
    cols, rows = dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1

    # load channels
    FLOAT = Imath.PixelType(Imath.PixelType.FLOAT)
    images = []
    for layer in layers:
        channels = LAYER_CHANNELS[layer]
        image = np.empty((rows, cols, 3), dtype=np.float32)
        for (i, c) in enumerate(channels):
            data = f.channel(c, FLOAT)
            image[:, :, i] = np.fromstring(data, dtype=np.float32) \
                .reshape((rows, cols))
        images.append(image)

    return images


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


def srgb_to_rgb(srgb):
    """ Convert an image from sRGB to linear RGB.

    :param srgb: numpy array in range (0.0 to 1.0)
    """
    ret = np.zeros_like(srgb)
    idx0 = srgb <= 0.04045
    idx1 = srgb > 0.04045
    ret[idx0] = srgb[idx0] / 12.92
    ret[idx1] = np.power((srgb[idx1] + 0.055) / 1.055, 2.4)
    return ret


def pil_to_numpy(pil):
    """ Convert an 8bit PIL image (0 to 255) to a floating-point numpy array
    (0.0 to 1.0) """
    return np.asarray(pil).astype(float) / 255.0


def numpy_to_pil(img):
    """ Convert a floating point numpy array (0.0 to 1.0) to an 8bit PIL image
    (0 to 255) """
    return Image.fromarray(
        np.clip(img * 255, 0, 255).astype(np.uint8)
    )

