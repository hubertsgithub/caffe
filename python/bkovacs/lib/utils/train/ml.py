import random


def split_train_test(data, test_ratio, seed=42):
    '''
    Splits an iterable data into two disjunct sets: training and test, the split is randomized
    '''
    random.seed(seed)
    testset_size = int(len(data) * test_ratio)

    if type(data) is dict:
        test_set_keys = random.sample(data.keys(), testset_size)
        test_set = {k: v for k, v in data.iteritems() if k in test_set_keys}
        train_set = {k: v for k, v in data.iteritems() if k not in test_set_keys}
    else:
        test_set = random.sample(data, testset_size)
        train_set = [x for x in data if x not in test_set]

    return train_set, test_set

