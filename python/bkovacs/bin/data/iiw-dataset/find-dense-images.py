import os
from os import listdir
import json

from lib.utils.misc.pathresolver import acrp

# this script finds all the dense images and writes their name to a file
origpath = acrp('data/iiw-dataset/data')
outputfilepath_dense = acrp('data/iiw-dataset/denseimages.txt')
outputfilepath_others = acrp('data/iiw-dataset/all-except-denseimages.txt')
outputfilepath_all = acrp('data/iiw-dataset/all-images.txt')

origdirnames = listdir(origpath)
# filter for only json files
origdirnames = [x for x in origdirnames if os.path.splitext(x)[1] == '.json']
origdirnames.sort()

denseimgcount = 0
othersimgcount = 0
distximgcount = 0
distyimgcount = 0
incompleteimgs = set()

with open(outputfilepath_dense, 'w') as dense_file, open(outputfilepath_others, 'w') as others_file, open(outputfilepath_all, 'w') as all_file:
    for filename in origdirnames:
        print 'Processing file {0}...'.format(filename)

        filepath = os.path.join(origpath, filename)
        trunc_filename, ext = os.path.splitext(filename)

        if not os.path.exists(os.path.join(origpath, '{0}-dist_x.png'.format(trunc_filename))):
            distximgcount += 1
            incompleteimgs.add(trunc_filename)

        if not os.path.exists(os.path.join(origpath, '{0}-dist_y.png'.format(trunc_filename))):
            distyimgcount += 1
            incompleteimgs.add(trunc_filename)

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

        all_file.write(trunc_filename + '\n')

print 'Found {0} dense images'.format(denseimgcount)
print 'Found {0} other images'.format(othersimgcount)
print '{0} images altogether'.format(denseimgcount + othersimgcount)
print '{0} distx and {1} disty missing images'.format(distximgcount, distyimgcount)
print 'Incomplete images: {0}'.format(incompleteimgs)

print 'Done.'







