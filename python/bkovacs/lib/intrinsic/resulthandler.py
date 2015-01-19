import redis
import re
import numpy as np
import time
import os

from lib.utils.data import common, fileproc
from lib.utils.misc import packer, strhelper
from lib.utils.misc.progressbaraux import progress_bar_widgets
from progressbar import ProgressBar
from celeryconfig_local import PASSWORD, REDISIP

REDIS_CONFIG = {'host': REDISIP, 'port': 6379, 'password': PASSWORD, 'db': 0}
#REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'password': None, 'db': 0}
BATCHSIZE = 128
PATTERN_INTERMEDIARY = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
PATTERN_IMAGERESULTS = 'intrinsicresults-imageAsResult-class=([^-]*)-tag=([^-]*)-classnum=([^-]*)-samplenum=([^-]*)'


def get_task_key(prefix, EstimatorClass, tag, jobid_params):
    return 'intrinsicresults-{0}-class={1}-tag={2}-{3}'.format(prefix, EstimatorClass, tag, strhelper.dicstr(jobid_params))


def delete_results(keys):
    client = redis.StrictRedis(**REDIS_CONFIG)
    client.delete(keys)


def computeScoreJob_sendresults(*args, **kwargs):
    from lib.intrinsic.comparison import computeScoreJob
    key, value = computeScoreJob(*args, **kwargs)

    print 'Starting redis client...'
    client = redis.StrictRedis(**REDIS_CONFIG)
    print 'Setting key: {0}'.format(key)
    client.set(key, value)


def get_all_processed():
    client = redis.StrictRedis(**REDIS_CONFIG)
    all_processed = set()
    for key in client.scan_iter('intrinsicresults-intermediary-class=*'):
        all_processed.add(key)

    return all_processed


def wait_all_results(nchoices_forclass, ntags):
    client = redis.StrictRedis(**REDIS_CONFIG)
    readydict = {EstimatorClass: False for EstimatorClass in nchoices_forclass}
    allready = False

    allcount = 0
    for nchoices in nchoices_forclass.itervalues():
        allcount += nchoices

    allcount *= ntags
    print 'Waiting for {0} jobs to complete'.format(allcount)

    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=allcount)
    pbar.start()
    pbar_counter = 0
    visited = set()

    while not allready:
        allready = True
        for EstimatorClass, nchoices in nchoices_forclass.iteritems():
            if readydict[EstimatorClass]:
                continue

            allready = False
            count = 0
            for key in client.scan_iter('intrinsicresults-intermediary-class={0}*'.format(EstimatorClass)):
                count += 1

                # Don't show the progress for a key twice
                if key in visited:
                    continue

                pbar_counter += 1
                pbar.update(pbar_counter)
                visited.add(key)

            print '{0} progress: {1}/{2}'.format(EstimatorClass, count, nchoices * ntags)

            if count == nchoices * ntags:
                readydict[EstimatorClass] = True
                print '{0} READY'.format(EstimatorClass)

        time.sleep(5)

    pbar.finish()

    print 'ALL READY!'


def get_scores_by_keys(scores, keys_to_get, client, pattern, visited):
    if len(keys_to_get) == 0:
        return

    keys_to_get = list(keys_to_get)
    values = client.mget(keys_to_get)

    for key_idx, key in enumerate(keys_to_get):
        match = re.search(pattern, key)
        #print 'IntermedResGather, parsed key: {0}, results: {1}'.format(key, match.groups())
        estclass, tag, i, j = match.groups()
        i = int(i)
        j = int(j)

        packed = values[key_idx]
        version, value = packer.unpackb(packed)
        visited.add(key)

        # if the value is NaN, the task failed with LinAlgError
        if np.isnan(value):
            print 'Task {0} failed with LinAlgError (probably SVD did not converge)'.format(key)

        scores[i, j] = value


def gather_intermediary_results(EstimatorClass, ntags, nchoices, PARTIALRESULTS):
    scores = np.empty((ntags, nchoices))
    scores.fill(np.nan)
    remaining = ntags * nchoices

    client = redis.StrictRedis(**REDIS_CONFIG)
    visited = set()

    if PARTIALRESULTS:
        print 'Gathering intermediate partial results...'
    else:
        print 'Gathering intermediate results...'

    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=ntags * nchoices)
    pbar.start()
    pbar_counter = 0
    while remaining > 0:
        keys_to_get = set()
        for key in client.scan_iter('intrinsicresults-intermediary-class={0}*'.format(EstimatorClass)):
            if key in visited:
                continue

            keys_to_get.add(key)
            if len(keys_to_get) >= BATCHSIZE:
                get_scores_by_keys(scores, keys_to_get, client, PATTERN_INTERMEDIARY, visited)
                remaining -= len(keys_to_get)
                pbar_counter += len(keys_to_get)
                pbar.update(pbar_counter)
                keys_to_get.clear()

        # We might have some unprocessed keys
        get_scores_by_keys(scores, keys_to_get, client, PATTERN_INTERMEDIARY, visited)
        remaining -= len(keys_to_get)
        pbar_counter += len(keys_to_get)
        pbar.update(pbar_counter)
        keys_to_get.clear()

        # If we want only partial results, we don't wait for the other results to appear
        if PARTIALRESULTS:
            break

    pbar.finish()

    return scores


def get_all_processed_from_file(processed_tasks_filepath):
    if os.path.exists(processed_tasks_filepath):
        return fileproc.freadlines(processed_tasks_filepath)
    else:
        return []


def save_images_by_keys(keys_to_get, client, pattern, visited, results_dir, processed_tasks_filepath):
    if len(keys_to_get) == 0:
        return

    keys_to_get = list(keys_to_get)
    values = client.mget(keys_to_get)

    for key_idx, key in enumerate(keys_to_get):
        match = re.search(pattern, key)
        #print 'IntermedResGather, parsed key: {0}, results: {1}'.format(key, match.groups())
        estclass, tag, classnum, samplenum = match.groups()
        classnum = int(classnum)
        samplenum = int(samplenum)

        packed = values[key_idx]
        version, value = packer.unpackb(packed)
        visited.add(key)
        score, est_shading, est_refl = value

        # if the value is NaN, the task failed with LinAlgError
        if np.isnan(score):
            print 'Task {0} failed with LinAlgError (probably SVD did not converge)'.format(key)
        else:
            est_shading /= np.max(est_shading)
            est_shading = est_shading ** 1./2.2
            est_refl /= np.max(est_refl)
            est_refl = est_refl ** 1./2.2

            # save each tag into different directory
            tagdir = os.path.join(results_dir, tag)
            if not os.path.exists(tagdir):
                os.mkdir(tagdir)

            shading_file = '{0}-classnum{1}-samplenum{2}-shading.png'.format(tag, classnum, samplenum)
            refl_file = '{0}-classnum{1}-samplenum{2}-refl.png'.format(tag, classnum, samplenum)
            common.save_image(os.path.join(results_dir, tag, shading_file), est_shading, is_srgb=False)
            common.save_image(os.path.join(results_dir, tag, refl_file), est_refl, is_srgb=False)

    # Save progressed keys
    processed_keys = get_all_processed_from_file(processed_tasks_filepath)
    processed_keys += keys_to_get
    fileproc.fwritelines(processed_tasks_filepath, processed_keys)

    client.delete(keys_to_get)


def gather_all_jobresults(job_params, results_dir, processed_tasks_filepath):
    client = redis.StrictRedis(**REDIS_CONFIG)

    allcount = len(job_params)
    remaining = allcount
    print 'Waiting for {0} jobs to complete'.format(allcount)

    pbar = ProgressBar(widgets=progress_bar_widgets(), maxval=allcount)
    pbar.start()
    pbar_counter = 0
    visited = set()

    while remaining > 0:
        keys_to_get = set()
        for key in client.scan_iter('intrinsicresults-imageAsResult-*'):
            if key in visited:
                continue

            keys_to_get.add(key)
            if len(keys_to_get) >= BATCHSIZE:
                save_images_by_keys(keys_to_get, client, PATTERN_IMAGERESULTS, visited, results_dir, processed_tasks_filepath)
                remaining -= len(keys_to_get)
                pbar_counter += len(keys_to_get)
                pbar.update(pbar_counter)
                keys_to_get.clear()

        # We might have some unprocessed keys
        save_images_by_keys(keys_to_get, client, PATTERN_IMAGERESULTS, visited, results_dir, processed_tasks_filepath)
        remaining -= len(keys_to_get)
        pbar_counter += len(keys_to_get)
        pbar.update(pbar_counter)
        keys_to_get.clear()

    pbar.finish()

    print 'ALL READY!'


#def gather_final_results(EstimatorClass, ntags):
#    res = [() for i in range(ntags)]
#    remaining = ntags
#
#    client = redis.StrictRedis(**REDIS_CONFIG)
#    visited = set()
#    while remaining > 0:
#        keys_to_delete = set()
#        for key in client.scan_iter('intrinsicresults-final-class={0}*'.format(EstimatorClass)):
#            keys_to_delete.add(key)
#            if key in visited:
#                continue
#
#            match = re.search(PATTERN_INTERMEDIARY, key)
#            print 'FinalResGather, parsed key: {0}, results: {1}'.format(key, match.groups())
#            estclass, tag, i, j = match.groups()
#            i = int(i)
#            j = int(j)
#
#            packed = client.get(key)
#            version, value = packer.unpackb(packed)
#            visited.add(key)
#
#            res[i] = value
#            remaining -= 1
#
#        if keys_to_delete:
#            client.delete(*keys_to_delete)
#
#    return res
