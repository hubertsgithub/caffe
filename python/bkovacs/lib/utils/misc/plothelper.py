import os
import numpy as np
import matplotlib.pyplot as plt

from lib.utils.data import common

def plot_and_save_2D_arrays(filename, arrs, xlabel='', xinterval=None, ylabel='', yinterval=None, line_names=[]):
    for arr in arrs:
        if arr.ndim != 2 or arr.shape[1] != 2:
            raise ValueError('The array should be 2D and the second dimension should be 2!')

        plt.plot(arr[:, 0], arr[:, 1])

    name, ext = os.path.splitext(os.path.basename(filename))
    plt.title(name)
    plt.xlabel(xlabel)
    if xinterval:
        plt.xlim(xinterval)

    plt.ylabel(ylabel)
    if yinterval:
        plt.ylim(yinterval)

    if line_names:
        plt.legend(line_names, loc='best')

    plt.savefig(filename)
    plt.clf()


def plot_and_save_2D_array(filename, arr, xlabel='', xinterval=None, ylabel='', yinterval=None):
    plot_and_save_2D_arrays(filename, [arr], xlabel, xinterval, ylabel, yinterval, line_names=[])


# take an array of shape output_channels, height, width, input_channels
# and visualize each (height, width) thing in a grid of size approx. sqrt(n) by sqrt(n)
def vis_square(data, padsize=1):
    padval = -1
    data -= data.min()
    data /= data.max()

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


def save_vis_square(filename, data, padsize=1):
    padded_data = vis_square(data, padsize)

    common.save_image(filename, padded_data, is_srgb=False)
