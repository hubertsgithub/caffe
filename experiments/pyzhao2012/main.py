import sys
import os

sys.path.append('experiments')
sys.path.append('data')
import common
import poisson
import numpy as np
import scipy as sp

def main():
    rootpath = 'experiments/pyzhao2012'
    img = common.load_png(os.path.join(rootpath, 'diffuse.png'))
    mask = common.load_png(os.path.join(rootpath, 'mask.png')) > 0
    common.save_png(img, os.path.join(rootpath, 'img.png'))

    shading, reflectance = runzhao(img, mask)
    common.print_array_info(shading)
    common.print_array_info(reflectance)

    common.save_png(shading, os.path.join(rootpath, 'shading.png'))
    common.save_png(shading, os.path.join(rootpath, 'reflectance.png'))


def runzhao(img, mask):
    '''
    Input
    img: W x H x 3
        color image in linear RGB colorspace, values in [0.0, 1.0]
    '''

    threshold = 0.005
    chromimg = common.compute_chromaticity_image(img)
    chromimg = np.mean(chromimg, axis=2)
    grayimg = np.mean(img, axis=2)

    approx_max = np.percentile(grayimg, 99.9)
    max_inds = np.transpose(np.nonzero(grayimg >= approx_max))
    common.print_array_info(max_inds)

    log_chromimg = np.where(mask, np.log(grayimg), 0.)

    A, b, c = buildMatrices(log_chromimg, mask, threshold, max_inds)
    common.print_array_info(A)
    common.print_array_info(b)
    print c

    log_shading = sp.sparse.linalg.spsolve(A, b)
    log_shading = np.reshape(log_shading, chromimg.shape)
    shading = mask * np.exp(log_shading)

    return shading, np.where(mask, grayimg / shading, 0.)

def buildMatrices(log_chromimg, mask, threshold, max_inds):
    (width, height) = log_chromimg.shape
    resolution = width * height
    print 'width: {0}, height: {1}, resolution: {2}'.format(width, height, resolution)
    A = sp.sparse.lil_matrix((resolution, resolution))
    b = np.zeros(resolution)
    c = 0.0

    # handle neigbor cases
    for h in range(height):
        print 'row {0}...'.format(h)
        for w in range(width):
            if not mask[w, h]:
                continue

            p = h * w

            # go through all neighbors
            for dh in range(-1, 1, 2):
                for dw in range(-1, 1, 2):
                    ch = h + dh
                    cw = w + dw

                    if ch < 0 or cw < 0 or ch >= height or cw >= width:
                        continue

                    dI = log_chromimg[w][h] - log_chromimg[cw][ch]
                    weight = computeWeight(dI, threshold)
                    q = ch * cw

                    A[p, p] += 1 + weight
                    A[p, q] += -2 * (1 + weight)
                    A[q, q] += 1 + weight

                    b[p] += -2 * weight * dI
                    b[q] += 2 * weight * dI

                    c += weight * dI * dI

    for i in max_inds:
        p = i[0] * i[1]
        A[p, p] += 1
        b[p] += -2
        c += 1

    b *= -1
    A *= 2

    return (A, b, c)


def computeWeight(dI, threshold):
    dist = dI * dI

    if dist > threshold:
        return 0
    else:
        return 100


main()
