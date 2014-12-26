import os

import math
import numpy as np
import scipy as sp
from scipy import optimize
from scipy import spatial
import itertools
import pyamg
import random

from lib.utils.data import common
from lib.utils.misc.pathresolver import acrp

ROOTPATH = acrp('experiments/pyzhao2012/input')

#def main():
#    if MITINTRINSIC:
#        smalladd = ''  # '-converted'
#        img = common.load_png(os.path.join(ROOTPATH, 'diffuse{0}.png'.format(smalladd)))
#        mask = common.load_png(os.path.join(ROOTPATH, 'mask{0}.png'.format(smalladd))) > 0
#        groups = None
#    else:
#        img = common.load_image(os.path.join(ROOTPATH, '662.png'), is_srgb=True)
#        width, height = img.shape[0:2]
#        mask = np.ones((width, height))
#        judgements = json.load(open(os.path.join(ROOTPATH, '662.json')))
#        groups = findIIWGroups(judgements, width, height, THRESHOLD_CONFIDENCE, GROUP_WEIGHT)
#        #groups = []
#
#    shading, reflectance = run(img, mask, groups)
#    reflectance = common.compute_color_reflectance(reflectance, img)
#    common.print_array_info(shading, 'final shading')
#    common.print_array_info(reflectance, 'final reflectance')
#
#    # gamma correction
#    shading = common.rgb_to_srgb(shading)
#    reflectance = common.rgb_to_srgb(reflectance)
#
#    common.save_png(shading, os.path.join(ROOTPATH, 'shading.png'))
#    common.save_png(reflectance, os.path.join(ROOTPATH, 'reflectance.png'))


def run(img, mask, lambda_l, lambda_r, lambda_a, abs_const_val, threshold_group_sim, threshold_chrom, groups=None):
    '''
    Input
    img: W x H x 3
        color image in linear RGB colorspace, values in [0.0, 1.0]

    Output
    A tuple: (shading, reflectance), both grayscale, linear RGB colorspace, values in [0.0, 1.0]
    '''

    chromimg = common.compute_chromaticity_image(img)
    grayimg = np.mean(img, axis=2)

    approx_max = np.percentile(grayimg, 99.9)
    max_inds = np.transpose(np.nonzero(grayimg >= approx_max))
    common.print_array_info(max_inds, 'max_inds')

    used_indtuple = np.nonzero(mask)
    used_indlist = np.transpose(used_indtuple)
    used_pxcount = used_indlist.shape[0]
    print 'Unmasked pixel count: {0}'.format(used_pxcount)
    used_inddic = dict([(tuple(x), i) for i, x in enumerate(used_indlist)])

    log_grayimg = np.where(mask, np.log(np.clip(grayimg, 0.0001, np.infty)), 0.)

    computeRetinexContour(log_grayimg, chromimg, mask, threshold_chrom, max_inds)

    #x0 = np.zeros(used_pxcount)
    #x0.fill(0)
    #res = optimize.minimize(method='L-BFGS-B', fun=func, x0=x0, jac=func_deriv, args=(log_grayimg, chromimg, used_inddic, lambda_l, lambda_a, abs_const_val, threshold_chrom, max_inds), options={'disp': True})
    #log_shading = np.zeros_like(grayimg)
    #log_shading[used_indtuple] = res.x

    if groups is None:
        groups = []
        #groups = findGroups(chromimg, used_indlist, 3, GROUP_SIM_THRESHOLD)

    grouped_pxcount = sum(1 for g in groups for px in g)
    print 'Number of groups: {0} / {1} pixels'.format(len(groups), used_pxcount)
    print 'Number of grouped pixels: {0} / {1} pixels'.format(grouped_pxcount, used_pxcount)

    A, b, c = buildMatrices(log_grayimg, chromimg, used_inddic, used_pxcount, groups, lambda_l, lambda_r, lambda_a, abs_const_val, threshold_chrom, max_inds)

    #res = sp.sparse.linalg.spsolve(A, b)
    solver = pyamg.ruge_stuben_solver(A)
    res = solver.solve(b)
    log_shading = np.zeros_like(grayimg)
    log_shading[used_indtuple] = res
    common.print_array_info(log_shading, 'log_shading')

    shading = mask * np.exp(log_shading)
    common.print_array_info(shading, 'shading')
    shading = shading / np.max(shading)
    refl = np.where(mask, grayimg / shading, 0.)
    refl = refl / np.max(refl)

    return shading, refl


def computeRetinexContour(log_grayimg, chromimg, mask, threshold_chrom, max_inds):
    (width, height) = log_grayimg.shape
    contourimg = np.ones_like(log_grayimg)

    # Retinex constraint
    for h in range(height):
        for w in range(width):
            if not mask[w, h]:
                continue

            # go through all neighbors
            neighbors = [[h-1, w], [h+1, w], [h, w-1], [h, w+1]]
            for ch, cw in neighbors:
                if ch < 0 or cw < 0 or ch >= height or cw >= width:
                    continue

                weight = computeWeight(chromimg, w, h, cw, ch, threshold_chrom)
                if weight == 0.:
                    contourimg[w, h] = 0.

    contourimg = contourimg * mask
    #common.save_png(contourimg, os.path.join(ROOTPATH, 'contourimg.png'))


def findGroups(chromimg, used_indlist, window_size, threshold_group_sim):
    # TODO: add group confidence and choose the group with the highest confidence for each pixel
    (width, height, _) = chromimg.shape
    # True if matched, initially none of them are matched (all False)
    matched_px = np.zeros(len(used_indlist), dtype=bool)
    groups = []

    # N * K array, N = number of used pixels, K = feature dimension, it's 3x3=9 here
    features = [getWindow(chromimg, w, h, window_size).flatten() for w, h in used_indlist]

    tree = spatial.cKDTree(features)

    for i, (w, h) in enumerate(used_indlist):
        if matched_px[i]:
            continue

        print 'Searching neighbors for the {0}th pixel'.format(i)
        window = getWindow(chromimg, w, h, window_size).flatten()
        # search all neighbors of window in a group_sim_threshold radius ball
        neighbor_inds = tree.query_ball_point(window, threshold_group_sim)
        neighbor_inds.append(0)

        newgroup = used_indlist[neighbor_inds]
        matched_px[neighbor_inds] = True

        # if the group is not only the first pixel, we add it to the similarity groups
        if len(newgroup) > 1:
            groups.append(newgroup)

    return groups


def getWindow(chromimg, w, h, window_size):
    window_shift = (window_size-1)/2

    return chromimg[(w-window_shift):(w+window_shift), (h-window_shift):(h+window_shift)]


def findIIWGroups(judgements, width, height, threshold_confidence):
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}

    cur_groupid = 0
    pointid_groupid = {}
    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # we are interested only in point pairs with 'equal' reflectance
        if darker != 'E':
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        # if the confidence is not high enough, skip point
        if weight < threshold_confidence:
            continue

        pointid1 = c['point1']
        pointid2 = c['point2']
        point1 = id_to_points[pointid1]
        point2 = id_to_points[pointid2]
        if not point1['opaque'] or not point2['opaque']:
            continue

        if pointid1 in pointid_groupid:
            pointid_groupid[pointid2] = pointid_groupid[pointid1]
        elif pointid2 in pointid_groupid:
            pointid_groupid[pointid1] = pointid_groupid[pointid2]
        else:
            # create new group
            pointid_groupid[pointid1] = pointid_groupid[pointid2] = cur_groupid
            cur_groupid += 1

    # go through the dictionary and create groups
    groupid_pointid = {}
    for pointid, groupid in pointid_groupid.iteritems():
        if groupid not in groupid_pointid:
            groupid_pointid[groupid] = []

        groupid_pointid[groupid].append(pointid)

    groups = []
    # create final groups
    for groupid, pointlist in groupid_pointid.iteritems():
        g = []
        for pid in pointlist:
            point = id_to_points[pid]
            w = int(point['x'] * width)
            h = int(point['y'] * height)

            g.append([w, h])

        groups.append(g)

    return groups


def buildMatrices(log_grayimg, chromimg, used_inddic, used_pxcount, groups, lambda_l, lambda_r, lambda_a, abs_const_val, threshold_chrom, max_inds):
    (width, height) = log_grayimg.shape
    A = sp.sparse.lil_matrix((used_pxcount, used_pxcount))
    b = np.zeros(used_pxcount)
    c = 0.0

    print 'Adding matrix elements for Retinex constraint'
    # Retinex constraint
    for h in range(height):
        for w in range(width):
            if (w, h) not in used_inddic:
                continue

            p = used_inddic[(w, h)]

            # go through all neighbors
            neighbors = [[h-1, w], [h+1, w], [h, w-1], [h, w+1]]
            for ch, cw in neighbors:
                if (cw, ch) not in used_inddic:
                    continue

                dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                weight = computeWeight(chromimg, w, h, cw, ch, threshold_chrom)
                q = used_inddic[(cw, ch)]

                A[p, p] += (1 + weight) * lambda_l
                A[p, q] += (-(1 + weight)) * lambda_l
                A[q, p] += (-(1 + weight)) * lambda_l
                A[q, q] += (1 + weight) * lambda_l

                b[p] += (-2 * weight * dI) * lambda_l
                b[q] += (2 * weight * dI) * lambda_l

                c += (weight * dI * dI) * lambda_l

    print 'Adding matrix elements for non-local texture constraint'
    # non-local texture constraint
    for i, g in enumerate(groups):
        print 'Group {0} with {1} elements'.format(i, len(g))
        sample_count = 200
        sampling_weight = 1.0
        if len(g) > sample_count:
            # if we have too many groups, sample randomly
            sampling_weight = float(len(g)) / sample_count
            g = random.sample(g, sample_count)

        for p_coord, q_coord in itertools.combinations(g, 2):
            w, h = p_coord
            cw, ch = q_coord
            p = used_inddic[(w, h)]
            q = used_inddic[(cw, ch)]
            dI = log_grayimg[w][h] - log_grayimg[cw][ch]

            A[p, p] += 1 * lambda_r * sampling_weight
            A[p, q] += -1 * lambda_r * sampling_weight
            A[q, p] += -1 * lambda_r * sampling_weight
            A[q, q] += 1 * lambda_r * sampling_weight

            b[p] += (-2 * dI) * lambda_r * sampling_weight
            b[q] += (2 * dI) * lambda_r * sampling_weight

            c += (dI * dI) * lambda_r * sampling_weight

    print 'Adding matrix elements for absolute scale constraint'
    # absolute scale constraint
    for i in max_inds:
        p = used_inddic[(i[0], i[1])]
        A[p, p] += 1 * lambda_a
        b[p] += -2 * abs_const_val * lambda_a
        c += abs_const_val ** 2 * lambda_a

    b *= -1
    A *= 2

    return (A, b, c)


def computeWeight(chromimg, w, h, cw, ch, threshold_chrom):
    dRhat = chromimg[w][h] - chromimg[cw][ch]
    dist = math.sqrt(np.sum(np.power(dRhat, 2.)))

    if dist > threshold_chrom:
        return 0.
    else:
        return 100.


def func(s, log_grayimg, chromimg, used_inddic, lambda_l, lambda_a, abs_const_val, threshold_chrom, max_inds):
    (width, height) = log_grayimg.shape
    sum = 0.0

    # Retinex constraint
    for h in range(height):
        for w in range(width):
            if (w, h) not in used_inddic:
                continue

            p = used_inddic[(w, h)]

            # go through all neighbors
            neighbors = [[h-1, w], [h+1, w], [h, w-1], [h, w+1]]
            for ch, cw in neighbors:
                if (cw, ch) not in used_inddic:
                    continue

                dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                weight = computeWeight(chromimg, w, h, cw, ch, threshold_chrom)
                q = used_inddic[(cw, ch)]

                dS = s[p] - s[q]
                sum += (dS ** 2 + weight * (dI - dS) ** 2) * lambda_l

    # absolute scale constraint
    for i in max_inds:
        p = used_inddic[(i[0], i[1])]
        sum += (s[p] - abs_const_val) ** 2 * lambda_a

    return sum


def func_deriv(s, log_grayimg, chromimg, used_inddic, lambda_l, lambda_a, abs_const_val, threshold_chrom, max_inds):
    (width, height) = log_grayimg.shape
    grad = np.zeros_like(s)

    # Retinex constraint
    for h in range(height):
        for w in range(width):
            if (w, h) not in used_inddic:
                continue

            p = used_inddic[(w, h)]

            sum = 0.0

            # go through all neighbors
            neighbors = [[h-1, w], [h+1, w], [h, w-1], [h, w+1]]
            for ch, cw in neighbors:
                if (cw, ch) not in used_inddic:
                    continue

                dI = log_grayimg[w][h] - log_grayimg[cw][ch]
                weight = computeWeight(chromimg, w, h, cw, ch, threshold_chrom)
                q = used_inddic[(cw, ch)]

                # * 2 at the end, because we compute the same for all neighbors and (a - b)^2 and (b - a)^2 are the same
                sum += (2 * (1 + weight) * s[p] + s[q] * (-2 * (1 + weight)) - 2 * weight * dI) * lambda_l * 2

            grad[p] += sum

    # absolute scale constraint
    for i in max_inds:
        p = used_inddic[(i[0], i[1])]
        grad[p] += 2 * (s[p] - abs_const_val) * lambda_a

    return grad


