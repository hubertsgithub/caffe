import os
from os import listdir
import json

from lib.utils.misc.pathresolver import acrp

# this script finds all the dense images and writes their name to a file
origpath = acrp('data/iiw-dataset/data')
outputfilepath_dense = acrp('data/iiw-dataset/denseimages.txt')
outputfilepath_others = acrp('data/iiw-dataset/all-except-denseimages.txt')

origdirnames = listdir(origpath)
# filter for only json files
origdirnames = [x for x in origdirnames if os.path.splitext(x)[1] == '.json']
origdirnames.sort()

denseimgcount = 0
othersimgcount = 0

with open(outputfilepath_dense, 'w') as dense_file, open(outputfilepath_others, 'w') as others_file:
    for filename in origdirnames:
        print 'Processing file {0}...'.format(filename)

        filepath = os.path.join(origpath, filename)
        trunc_filename, ext = os.path.splitext(filename)

        judgements = json.load(open(filepath))
        points = judgements['intrinsic_points']

        if len(points) == 0:
            continue

        found = False
        for p in points:
            if p['min_separation'] == 0.03:
                found = True
                break

        if found:
            dense_file.write(trunc_filename + '\n')
            denseimgcount += 1
        else:
            others_file.write(trunc_filename + '\n')
            othersimgcount += 1


print 'Found {0} dense images'.format(denseimgcount)
print 'Found {0} other images'.format(othersimgcount)


print 'Done.'







