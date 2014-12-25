import redis
import re
import numpy as np

from lib.utils.misc import packer

REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'password': None, 'db': 0}


def computeScoreJob_sendresults(*args, **kwargs):
    from lib.intrinsic.comparison import computeScoreJob
    key, value = computeScoreJob(*args, **kwargs)

    print 'Starting redis client...'
    client = redis.StrictRedis(**REDIS_CONFIG)
    print 'Setting key: {0}'.format(key)
    client.set(key, value)


def gather_intermediary_results(EstimatorClass, ntags, nchoices):
    scores = np.zeros((ntags, nchoices))
    remaining = ntags * nchoices

    pattern = 'intrinsicresults-intermediary-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    visited = set()
    while remaining > 0:
        keys_to_delete = set()
        for key in client.scan_iter('intrinsicresults-intermediary-class={0}*'.format(EstimatorClass)):
            keys_to_delete.add(key)
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

        if keys_to_delete:
            client.delete(*keys_to_delete)

    return scores


def gather_final_results(EstimatorClass, ntags):
    res = [() for i in range(ntags)]
    remaining = ntags

    pattern = 'intrinsicresults-final-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    visited = set()
    while remaining > 0:
        keys_to_delete = set()
        for key in client.scan_iter('intrinsicresults-final-class={0}*'.format(EstimatorClass)):
            keys_to_delete.add(key)
            if key in visited:
                continue

            match = re.search(pattern, key)
            print 'FinalResGather, parsed key: {0}, results: {1}'.format(key, match.groups())
            estclass, tag, i, j = match.groups()
            i = int(i)
            j = int(j)

            packed = client.get(key)
            version, value = packer.unpackb(packed)
            visited.add(key)

            res[i] = value
            remaining -= 1

        if keys_to_delete:
            client.delete(*keys_to_delete)

    return res
