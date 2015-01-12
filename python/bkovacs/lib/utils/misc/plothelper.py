import os
import matplotlib.pyplot as plt


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


