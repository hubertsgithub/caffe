import os
from os import listdir
import sys
import json
import random
import numpy as np
import cPickle as pickle
import multiprocessing

from lib.utils.data import common
from lib.utils.misc.pathresolver import acrp
from lib.utils.misc.parallel import call_with_multiprocessing_pool

g_SMALLERDIMSIZE = 1100
g_NETINPUTSIZE = 224
g_USESIMPLEGAMMAFORSAVE = True
g_MINDIST = g_NETINPUTSIZE / 2 + 1

# this script finds all the dense images and writes their name to a file
g_rootpath = acrp('data/iiw-dataset')
g_origpath = os.path.join(g_rootpath, 'data')
g_highresrootpath = acrp('../OpenSurfaces/photos')

g_equal_cmp_train = []
g_notequal_cmp_train = []
g_equal_cmp_test = []
g_notequal_cmp_test = []
g_skipped_count = 0
g_nofile_count = 0
g_processedfiles = set()
g_lock = multiprocessing.Lock()


def is_closeto_border(width, height, pw, ph, mindist):
    return pw - mindist < 0 or pw + mindist >= width or ph - mindist < 0 or ph + mindist >= height


def aggregator_func(ret):
    global g_equal_cmp_train, g_equal_cmp_test, g_notequal_cmp_train, g_notequal_cmp_test, g_skipped_count, g_nofile_count, g_processedfiles, g_lock

    g_lock.acquire()
    if 'filename' in ret:
        filename = ret['filename']
        g_processedfiles.add(filename)
        print '{0} processed'.format(filename)
        sys.stdout.flush()

        if 'equal_cmp' in ret:
            if filename in testset:
                equal_cmp = g_equal_cmp_test
                notequal_cmp = g_notequal_cmp_test
            else:
                equal_cmp = g_equal_cmp_train
                notequal_cmp = g_notequal_cmp_train

            equal_cmp.extend(ret['equal_cmp'])
            notequal_cmp.append(ret['notequal_cmp'])

        if 'nofile_count' in ret:
            g_nofile_count += 1

        if 'skipped_count' in ret:
            g_skipped_count += ret['skipped_count']

        dic = {'VERSION': 1.0, 'processedfiles': g_processedfiles, 'skipped_count': g_skipped_count, 'nofile_count': g_nofile_count}
        pickle.dump(dic, open(procfilepath, 'w'))

    g_lock.release()


def process_photos(pool, origdirnames, processedfiles, origpath, highresrootpath, SMALLERDIMSIZE, USESIMPLEGAMMAFORSAVE, MINDIST):
    for filename in origdirnames:
        if filename in processedfiles:
            print 'File already processed, skipping'
            continue

        res = pool.apply_async(process_photo, args=(filename, origpath, highresrootpath, SMALLERDIMSIZE, USESIMPLEGAMMAFORSAVE, MINDIST), callback=aggregator_func)


def process_photo(filename, origpath, highresrootpath, SMALLERDIMSIZE, USESIMPLEGAMMAFORSAVE, MINDIST):
    print 'Processing file {0}...'.format(filename)

    filepath = os.path.join(origpath, filename)
    trunc_filename, ext = os.path.splitext(filename)
    trunc_filepath, _ = os.path.splitext(filepath)
    highresimgfilepath = os.path.realpath(os.path.join(highresrootpath, trunc_filename + '.jpg'))

    if not os.path.exists(highresimgfilepath):
        print 'No highres path exists for {0}'.format(filename)
        return {'filename': filename, 'nofile_count': True}

    # load image, compute grayscale + chromaticity images
    linimg = common.load_image(highresimgfilepath, is_srgb=True)

    # the minimum resolution should be SMALLERDIMSIZE
    linimg = common.resize_and_crop_image(linimg, resize=SMALLERDIMSIZE, crop=None, keep_aspect_ratio=True, use_greater_side=False)
    width, height = linimg.shape[0:2]
    grayimg = np.mean(linimg, axis=2)
    chromimg = common.compute_chromaticity_image(linimg)

    # gamma correction
    if USESIMPLEGAMMAFORSAVE:
        grayimg = grayimg ** (1. / 2.2)
        chromimg = chromimg ** (1. / 2.2)
    else:
        grayimg = common.rgb_to_srgb(grayimg)
        chromimg = common.rgb_to_srgb(chromimg)

    grayimg_path = trunc_filepath + '-gray.png'
    chromimg_path = trunc_filepath + '-chrom.png'
    common.save_image(grayimg_path, grayimg, is_srgb=False)
    common.save_image(chromimg_path, chromimg, is_srgb=False)

    judgements = json.load(open(filepath))
    points = judgements['intrinsic_points']
    comparisons = judgements['intrinsic_comparisons']
    id_to_points = {p['id']: p for p in points}

    if not points:
        print 'Found no points for {0}'.format(filename)
        return {'filename': filename}

    equal_cmp = []
    notequal_cmp = []
    skipped_count = 0
    for c in comparisons:
        # "darker" is "J_i" in our paper
        darker = c['darker']
        if darker not in ('1', '2', 'E'):
            continue

        # "darker_score" is "w_i" in our paper
        weight = c['darker_score']
        if weight <= 0 or weight is None:
            continue

        point1 = id_to_points[c['point1']]
        point2 = id_to_points[c['point2']]
        if not point1['opaque'] or not point2['opaque']:
            continue

        p1x = point1['x']
        p1y = point1['y']
        p2x = point2['x']
        p2y = point2['y']

        p1w = p1x * width
        p1h = p1y * height
        p2w = p2x * width
        p2h = p2y * height

        # skip points which are close to the border of the image (so we can't crop)
        if is_closeto_border(width, height, p1w, p1h, MINDIST) or is_closeto_border(width, height, p2w, p2h, MINDIST):
            skipped_count += 1
            continue

        tup = (trunc_filepath, p1x, p1y, p2x, p2y)

        if darker == 'E':
            equal_cmp.append(tup)
        else:
            notequal_cmp.append(tup)

    print 'Successfully processed {0}!'.format(filename)
    return {'filename': filename, 'equal_cmp': equal_cmp, 'notequal_cmp': notequal_cmp, 'skipped_count': skipped_count}


if __name__ == "__main__":
    multiprocessing.log_to_stderr()
    origdirnames = listdir(g_origpath)
    # filter for only json files
    origdirnames = [x for x in origdirnames if os.path.splitext(x)[1] == '.json']
    origdirnames.sort()

    random.seed(42)
    # Select 20% test set randomly
    testset_size = int(len(origdirnames) * 0.2)
    trainset_size = len(origdirnames) - testset_size
    testset = random.sample(origdirnames, testset_size)

    f_train = open(os.path.join(g_rootpath, 'train.txt'), 'w')
    f_test = open(os.path.join(g_rootpath, 'test.txt'), 'w')

    procfilepath = os.path.join(g_rootpath, 'processedfiles.dat')
    if os.path.exists(procfilepath):
        dic = pickle.load(open(procfilepath, 'r'))

        VERSION = dic['VERSION']
        if VERSION != 1.0:
            raise ValueError('Wrong verison, please delete processedfiles.dat')

        processedfiles = dic['processedfiles']
        skipped_count = dic['skipped_count']
        nofile_count = dic['nofile_count']

    call_with_multiprocessing_pool(process_photos, origdirnames, g_processedfiles, g_origpath, g_highresrootpath, g_SMALLERDIMSIZE, g_USESIMPLEGAMMAFORSAVE, g_MINDIST)

    print 'Skipped {0} comparisons because they were close to the border'.format(g_skipped_count)
    print 'No highres file found for {0} images'.format(g_nofile_count)

    print 'Saving gathered info to training and testing files...'

    for f, equal_cmp, notequal_cmp in [(f_test, g_equal_cmp_test, g_notequal_cmp_test), (f_train, g_equal_cmp_train, g_notequal_cmp_train)]:
        print 'Number of equal comparisons found: {0}'.format(len(equal_cmp))
        print 'Number of notequal comparisons found: {0}'.format(len(notequal_cmp))
        for i, c in enumerate(equal_cmp):
            # for every positive example, we put a negative example too
            trunc_filename, p1x, p1y, p2x, p2y = c
            grayimg_path = os.path.join(g_origpath, trunc_filename) + '-gray.png'
            chromimg_path = os.path.join(g_origpath, trunc_filename) + '-chrom.png'
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(grayimg_path, chromimg_path, p1x, p1y, p2x, p2y))

            c = notequal_cmp[i % len(notequal_cmp)]
            trunc_filename, p1x, p1y, p2x, p2y = c
            grayimg_path = os.path.join(g_origpath, trunc_filename) + '-gray.png'
            chromimg_path = os.path.join(g_origpath, trunc_filename) + '-chrom.png'
            f.write('{0} {1} {2} {3} {4} {5}\n'.format(grayimg_path, chromimg_path, p1x, p1y, p2x, p2y))

    f_train.close()
    f_test.close()

    print 'Done.'







