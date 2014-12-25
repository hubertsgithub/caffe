import redis
import re
import numpy as np

REDIS_CONFIG = {'host': 'localhost', 'port': 6379, 'password': None, 'db': 0}


def computeScoreJob_sendresults(*args, **kwargs):
    from lib.intrinsic.comparison import computeScoreJob
    key, value = computeScoreJob(*args, **kwargs)

    print 'Starting redis client...'
    client = redis.StrictRedis(**REDIS_CONFIG)
    print 'Setting key: {0}, score: {1}'.format(key, value)
    client.set(key, value)


def gatherresults(EstimatorClass, ntags, nchoices):
    scores = np.zeros((ntags, nchoices))
    remaining = ntags * nchoices

    pattern = 'intrinsicresults-class=([^-]*)-tag=([^-]*)-i=([^-]*)-j=([^-]*)'
    client = redis.StrictRedis(**REDIS_CONFIG)
    visited = set()
    while remaining > 0:
        keys_to_delete = set()
        for key in client.scan_iter('intrinsicresults-class={0}*'.format(EstimatorClass)):
            keys_to_delete.add(key)
            if key in visited:
                continue

            match = re.search(pattern, key)
            print 'Parsed key: {0}, results: {1}'.format(key, match.groups())
            estclass, tag, i, j = match.groups()

            value = client.get(key)
            visited.add(key)

            scores[i, j] = value
            remaining -= 1

        if keys_to_delete:
            client.delete(*keys_to_delete)

    return scores

