import os
import StringIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def plot_2D_arrays(arrs, title='', xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[]):
    plt.clf()

    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('The array should be 2D and the second dimension should be 2!')

        plt.plot(arr[:, 0], arr[:, 1])

    plt.title(title)
    plt.xlabel(xlabel)
    if xinterval:
        plt.xlim(xinterval)

    plt.ylabel(ylabel)
    if yinterval:
        plt.ylim(yinterval)

    if line_names:
        plt.legend(line_names, loc='best')


def plot_and_svg_2D_arrays(filename, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[]):
    name, ext = os.path.splitext(os.path.basename(filename))
    plot_2D_arrays(arrs, name, xlabel, xinterval, ylabel, yinterval, line_names)

    buf = StringIO.StringIO()
    plt.savefig(buf, format='svg')
    plt.clf()

    return buf.getvalue()


def plot_and_save_2D_arrays(filename, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[]):
    name, ext = os.path.splitext(os.path.basename(filename))
    plot_2D_arrays(arrs, name, xlabel, xinterval, ylabel, yinterval, line_names)
    plt.savefig(filename)
    plt.clf()


def plot_and_save_2D_array(filename, arr, xlabel='', xinterval=None, ylabel='', yinterval=None):
    plot_and_save_2D_arrays(filename, [arr], xlabel, xinterval, ylabel, yinterval, line_names=[])


# take an array of shape output_channels, height, width, input_channels
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1):
    padval = -1
    data_min = data.min()
    data_max = data.max()

    data -= data_min
    if data_max > 1e-5:
        data /= data_max

    output_channels, input_channels, height, width = data.shape

    # tile the filters into an image
    data = np.pad(data, pad_width=((0, 0), (0, 0), (0, padsize), (0, padsize)), mode='constant', constant_values=(padval,))
    new_data = np.empty(((width + 1) * output_channels, (height + 1) * input_channels))
    for o in range(output_channels):
        for i in range(input_channels):
            for h in range(height + 1):
                for w in range(width + 1):
                    new_data[o * (width + 1) + w, i * (height + 1) + h] = data[o, i, h, w]

    # create a color image with green separators
    indices = new_data == padval
    new_data = new_data[:, :, np.newaxis].repeat(3, axis=2)
    new_data[indices] = [0, 1, 0]

    return new_data


def fig2data ( fig ):
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw ( )

    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.fromstring ( fig.canvas.tostring_argb(), dtype=np.uint8 )
    buf.shape = ( w, h,4 )

    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll ( buf, 3, axis = 2 )
    return buf


def fig2img ( fig ):
    """
    @brief Convert a Matplotlib figure to a PIL Image in RGBA format and return it
    @param fig a matplotlib figure
    @return a Python Imaging Library ( PIL ) image
    """
    # put the figure pixmap into a numpy array
    buf = fig2data ( fig )
    w, h, d = buf.shape
    return Image.fromstring( "RGBA", ( w ,h ), buf.tostring( ) )


def create_analytic_image(data, padsize=1):
    raveled = np.ravel(data)
    if np.any(np.isnan(raveled)):
        print 'NaN value in data, skipping NaN values...'

    plt.hist(raveled[~np.isnan(raveled)], bins=100)
    histimg = fig2img(plt.gcf())
    plt.clf()

    padded_data = vis_square(data, padsize)
    padded_data *= 255.0
    padded_data = np.asarray(padded_data, dtype=np.uint8)
    dataimg = Image.fromarray(padded_data)

    w, h = dataimg.size
    hw, hh = histimg.size

    blank_image = Image.new('RGB', (w + hw, max(h, hh)))
    blank_image.paste(dataimg, (0, 0))
    blank_image.paste(histimg, (w, 0))

    return blank_image
