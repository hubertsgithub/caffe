import redis
import re
import numpy as np
import time

from lib.utils.misc import packer

REDIS_CONFIG = {'host': '10.37.154.210', 'port': 6379, 'password': None, 'db': 0}
#REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'password': None, 'db': 0}


def computeScoreJob_sendresults(*args, **kwargs):
    from lib.intrinsic.comparison import computeScoreJob
    key, value = computeScoreJob(*args, **kwargs)

    print 'Starting redis client...'
    client = redis.StrictRedis(**REDIS_CONFIG)
    print 'Setting key: {0}'.format(key)
    client.set(key, value)


def get_all_processed():
    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    all_processed = set()
    for key in client.scan_iter('intrinsicresults-intermediary-class=*'):
        all_processed.add(key)

    return all_processed


def wait_all_results(nchoices_forclass, ntags):
    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    readydict = {EstimatorClass : False for EstimatorClass, _ in nchoices_forclass}
    allready = False
    while not allready:
        allready = True
        for EstimatorClass, nchoices in nchoices_forclass.iteritems():
            if readydict[EstimatorClass]:
                continue

            allready = False
            count = 0
            for key in client.scan_iter('intrinsicresults-intermediary-class={0}*'.format(EstimatorClass)):
                match = re.search(pattern, key)
                estclass, tag, i, j = match.groups()
                i = int(i)
                j = int(j)
                count += 1

            print '{0} progress: {1}/{2}'.format(EstimatorClass, count, nchoices * ntags)
            if count == nchoices * ntags:
                readydict[EstimatorClass] = True
                print '{0} READY'.format(EstimatorClass)

        time.sleep(5)

    print 'ALL READY!'


def gather_intermediary_results(EstimatorClass, ntags, nchoices):
    scores = np.zeros((ntags, nchoices))
    remaining = ntags * nchoices

    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    visited = set()
    while remaining > 0:
        for key in client.scan_iter('intrinsicresults-intermediary-class={0}*'.format(EstimatorClass)):
            if key in visited:
                continue

            match = re.search(pattern, key)
            print 'IntermedResGather, parsed key: {0}, results: {1}'.format(key, match.groups())
            estclass, tag, i, j = match.groups()
            i = int(i)
            j = int(j)

            packed = client.get(key)
            version, value = packer.unpackb(packed)
            visited.add(key)

            scores[i, j] = value
            remaining -= 1

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
