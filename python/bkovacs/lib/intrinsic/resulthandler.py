import redis
import re
import numpy as np
import time

from lib.utils.misc import packer
from lib.utils.misc.progressbaraux import progress_bar_widgets
from progressbar import ProgressBar
from celeryconfig_local import PASSWORD, REDISIP

REDIS_CONFIG = {'host': REDISIP, 'port': 6379, 'password': PASSWORD, 'db': 0}
#REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'password': None, 'db': 0}
BATCHSIZE = 1024


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
    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
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

    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
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
                get_scores_by_keys(scores, keys_to_get, client, pattern, visited)
                remaining -= len(keys_to_get)
                pbar_counter += len(keys_to_get)
                pbar.update(pbar_counter)
                keys_to_get.clear()

        # We might have some unprocessed keys
        get_scores_by_keys(scores, keys_to_get, client, pattern, visited)
        remaining -= len(keys_to_get)
        pbar_counter += len(keys_to_get)
        pbar.update(pbar_counter)
        keys_to_get.clear()

        # If we want only partial results, we don't wait for the other results to appear
        if PARTIALRESULTS:
            break

    pbar.finish()

    return scores


#def gather_final_results(EstimatorClass, ntags):
#    res = [() for i in range(ntags)]
#    remaining = ntags
#
#    pattern = 'intrinsicresults-final-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
#    client = redis.StrictRedis(**REDIS_CONFIG)
#    visited = set()
#    while remaining > 0:
#        keys_to_delete = set()
#        for key in client.scan_iter('intrinsicresults-final-class={0}*'.format(EstimatorClass)):
#            keys_to_delete.add(key)
#            if key in visited:
#                continue
#
#            match = re.search(pattern, key)
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
