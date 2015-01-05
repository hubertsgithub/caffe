import json
from collections import OrderedDict

from lib.utils.misc.pathresolver import acrp

if __name__ == '__main__':
    inputfilepath = acrp('data/iiw-decompositions/intrinsic-decompositions-export.json')
    outputfilepath = acrp('data/iiw-decompositions/error-each-dense-image.txt')

    jsondata = json.load(open(inputfilepath))
    bell_densecrf = jsondata[0]
    if bell_densecrf['slug'] != 'bell2014_densecrf':
        raise ValueError('Incorrect input file')

    decomps = bell_densecrf['intrinsic_images_decompositions']

    mean_errors = OrderedDict()

    mean_err_sum = 0.
    mean_err_count = 0.
    for d in decomps:
        if d['mean_dense_error'] == 'null':
            continue

        filename = d['photo_id']
        cur_mean_err = d['mean_error']
        mean_err_sum += cur_mean_err
        mean_err_count += 1

        mean_errors[filename] = cur_mean_err

    mean_err = mean_err_sum / mean_err_count

    with open(outputfilepath, 'w') as text_file:
        for p in mean_errors.iteritems():
            p = [str(x) for x in p]
            text_file.write(' '.join(p))

    print 'Mean error: {0}'.format(mean_err)



