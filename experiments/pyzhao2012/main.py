import sys
import os

sys.path.append('experiments')
sys.path.append('data')
import common
import math
import numpy as np
import scipy as sp
from scipy import optimize

ROOTPATH = 'experiments/pyzhao2012'

THRESHOLD = 0.25
LAMBDA_L = 1.
LAMBDA_A = 1000.

def main():
    smalladd = '-converted'
    img = common.load_png(os.path.join(ROOTPATH, 'diffuse{0}.png'.format(smalladd)))
    mask = common.load_png(os.path.join(ROOTPATH, 'mask{0}.png'.format(smalladd))) > 0
    common.save_png(img, os.path.join(ROOTPATH, 'img.png'))

    shading, reflectance = runzhao(img, mask)
    common.print_array_info(shading)
    common.print_array_info(reflectance)

    common.save_png(shading, os.path.join(ROOTPATH, 'shading.png'))
    common.save_png(shading, os.path.join(ROOTPATH, 'reflectance.png'))


def runzhao(img, mask):
    '''
    Input
    img: W x H x 3
        color image in linear RGB colorspace, values in [0.0, 1.0]
    '''

    threshold = THRESHOLD
    chromimg = common.compute_chromaticity_image(img)
    common.save_png(chromimg, os.path.join(ROOTPATH, 'chromimg.png'))

    grayimg = np.mean(img, axis=2)
    common.save_png(grayimg, os.path.join(ROOTPATH, 'grayimg.png'))

    approx_max = np.percentile(grayimg, 99)
    max_inds = np.transpose(np.nonzero(grayimg >= approx_max))
    common.print_array_info(max_inds, 'max_inds')

    log_grayimg = np.where(mask, np.log(np.clip(grayimg, 0.0001, np.infty)), 0.)

    computeRetinexContour(log_grayimg, chromimg, mask, threshold, max_inds)

    (width, height) = log_grayimg.shape
    resolution = width * height
    x0 = np.zeros(resolution)
    x0.fill(0.25)
    #(log_shading, minfunval, infodict) = optimize.fmin_l_bfgs_b(func=func, x0=x0, fprime=func_deriv, args=(log_grayimg, chromimg, mask, LAMBDA_L, LAMBDA_A, threshold, max_inds), iprint=0)
    res = optimize.minimize(method='L-BFGS-B', fun=func, x0=x0, jac=func_deriv, args=(log_grayimg, chromimg, mask, LAMBDA_L, LAMBDA_A, threshold, max_inds), options={'disp': True})
    log_shading = res.x

    #A, b, c = buildMatrices(log_grayimg, chromimg, mask, LAMBDA_L, LAMBDA_A, threshold, max_inds)
    #common.print_array_info(A)
    #common.print_array_info(b)
    #print c

    #evals = sp.sparse.linalg.eigsh(A, k=15, return_eigenvectors=False, which='SM')
    #print evals
    #print np.all(evals > 0)
    #log_shading = sp.sparse.linalg.spsolve(A, b)
    log_shading = np.reshape(log_shading, grayimg.shape)
    common.print_array_info(log_shading, 'log_shading')
    shading = mask * np.exp(log_shading)
    common.print_array_info(shading, 'shading')
    shading = shading / np.max(shading)

    return shading, np.where(mask, grayimg / shading, 0.)

def computeRetinexContour(log_grayimg, chromimg, mask, threshold, max_inds):
    (width, height) = log_grayimg.shape
    contourimg = np.ones_like(log_grayimg)

    # Retinex constraint
    for h in range(height):
        for w in range(width):
            if not mask[w, h]:
                continue

            # go through all neighbors
            for dh in range(-1, 1, 2):
                for dw in range(-1, 1, 2):
                    ch = h + dh
                    cw = w + dw

                    if ch < 0 or cw < 0 or ch >= height or cw >= width:
                        continue

                    weight = computeWeight(chromimg, w, h, cw, ch, threshold)
                    if weight == 0.:
                        contourimg[w, h] = 0.

    contourimg = contourimg * mask
    common.save_png(contourimg, os.path.join(ROOTPATH, 'contourimg.png'))


def func(s, log_grayimg, chromimg, mask, lambda_l, lambda_a, threshold, max_inds):
    (width, height) = log_grayimg.shape
    sum = 0.0

    # Retinex constraint
    for h in range(height):
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

                    dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                    weight = computeWeight(chromimg, w, h, cw, ch, threshold)
                    q = ch * cw

                    dS = s[p] - s[q]
                    sum += (dS ** 2 + weight * (dI - dS) ** 2) * lambda_l

    # absolute scale constraint
    for i in max_inds:
        p = i[0] * i[1]
        sum += (s[p] - 1) ** 2 * lambda_a

    return sum

def func_deriv(s, log_grayimg, chromimg, mask, lambda_l, lambda_a, threshold, max_inds):
    (width, height) = log_grayimg.shape
    resolution = width * height
    grad = np.zeros(resolution)

    # Retinex constraint
    for h in range(height):
        for w in range(width):
            #if not mask[w, h]:
            #    continue

            p = h * w

            sum = 0.0

            # go through all neighbors
            for dh in range(-1, 1, 2):
                for dw in range(-1, 1, 2):
                    ch = h + dh
                    cw = w + dw

                    if ch < 0 or cw < 0 or ch >= height or cw >= width:
                        continue

                    dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                    weight = computeWeight(chromimg, w, h, cw, ch, threshold)
                    q = ch * cw

                    # * 2 at the end, because we compute the same for all neigbors and (a - b)^2 and (b - a)^2 are the same
                    sum += (2 * (1 + weight) * s[p] + s[q] * (-2 * (1 + weight)) - 2 * weight * dI) * lambda_l * 2

            grad[p] += sum

    # absolute scale constraint
    for i in max_inds:
        p = i[0] * i[1]
        grad[p] += 2 * (s[p] - 1) * lambda_a

    return grad

def buildMatrices(log_grayimg, chromimg, mask, lambda_l, lambda_a, threshold, max_inds):
    (width, height) = log_grayimg.shape
    resolution = width * height
    print 'width: {0}, height: {1}, resolution: {2}'.format(width, height, resolution)
    A = sp.sparse.lil_matrix((resolution, resolution))
    b = np.zeros(resolution)
    c = 0.0

    # Retinex constraint
    for h in range(height):
        print 'row {0}...'.format(h)
        for w in range(width):
            #if not mask[w, h]:
            #    continue

            p = h * w

            # go through all neighbors
            for dh in range(-1, 1, 2):
                for dw in range(-1, 1, 2):
                    ch = h + dh
                    cw = w + dw

                    if ch < 0 or cw < 0 or ch >= height or cw >= width:
                        continue

                    dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                    weight = computeWeight(chromimg, w, h, cw, ch, threshold)

                    q = ch * cw

                    A[p, p] += (1 + weight) * lambda_l
                    A[p, q] += (-2 * (1 + weight)) * lambda_l
                    A[q, q] += (1 + weight) * lambda_l

                    b[p] += (-2 * weight * dI) * lambda_l
                    b[q] += (2 * weight * dI) * lambda_l

                    c += (weight * dI * dI) * lambda_l

    # absolute scale constraint
    for i in max_inds:
        p = i[0] * i[1]
        A[p, p] += 1 * lambda_a
        b[p] += -2 * lambda_a
        c += 1 * lambda_a

    b *= -1
    A *= 2

    return (A, b, c)


def computeWeight(chromimg, w, h, cw, ch, threshold):
    dRhat = chromimg[w][h] - chromimg[cw][ch]
    dist = math.sqrt(np.sum(np.power(dRhat, 2.)))

    if dist > threshold:
        return 0.
    else:
        return 100.


main()
