import os
from os import listdir
import json

from lib.utils.misc.pathresolver import acrp

# this script finds all the dense images and writes their name to a file
origpath = acrp('data/iiw-dataset/data')
outputfilepath = acrp('data/iiw-dataset/denseimages.txt')

origdirnames = listdir(origpath)
# filter for only json files
origdirnames = [x for x in origdirnames if os.path.splitext(x)[1] == '.json']
origdirnames.sort()

denseimgcount = 0

with open(outputfilepath, 'w') as text_file:
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

        if not found:
            continue

        text_file.write(trunc_filename + '\n')
        denseimgcount += 1


print 'Found {0} dense images'.format(denseimgcount)


print 'Done.'







