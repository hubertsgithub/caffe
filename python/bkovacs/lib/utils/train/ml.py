import random
import numpy as np


def split_dic(dic, split_sizes):
    assert len(dic) >= np.sum(split_sizes)

    keys = dic.keys()
    random.shuffle(keys)
    remaining_keys = keys

    ret = []
    for size in split_sizes:
        split_keys = remaining_keys[:size]
        remaining_keys = remaining_keys[size:]

        split = {k: v for k, v in dic.iteritems() if k in split_keys}
        ret.append(split)

    if remaining_keys:
        split = {k: v for k, v in dic.iteritems() if k in remaining_keys}
        ret.append(split)

    return ret


def split_arr(arr, split_sizes):
    assert len(arr) >= np.sum(split_sizes)

    random.shuffle(arr)
    remaining_arr = arr

    ret = []
    for size in split_sizes:
        split = remaining_arr[:size]
        remaining_arr = remaining_arr[size:]

        ret.append(split)

    if remaining_arr:
        ret.append(remaining_arr)

    return ret


def split_data(data, split_sizes):
    if type(data) is dict:
        splits = split_dic(data, split_sizes)
    else:
        splits = split_arr(data, split_sizes)

    return splits


def split_train_test(data, test_ratio, seed=42):
    '''
    Splits an iterable data into two disjunct sets: training and test, the split is randomized
    '''
    random.seed(seed)
    testset_size = int(len(data) * test_ratio)

    splits = split_data(data, [testset_size])

    # reverse to get train, test order
    return splits[::-1]


def split_train_val_test(data, val_ratio, test_ratio, seed=42):
    '''
    Splits an iterable data into validation disjoint sets: training and test, the split is randomized
    '''
    random.seed(seed)
    # Divide the whole set into test and training
    testset_size = int(len(data) * test_ratio)
    training_val_size = len(data) - testset_size

    # Divide the training set into validation and training
    valset_size = int(training_val_size * val_ratio)

    splits = split_data(data, [testset_size, valset_size])

    # reverse to get train, val, test order
    return splits[::-1]

