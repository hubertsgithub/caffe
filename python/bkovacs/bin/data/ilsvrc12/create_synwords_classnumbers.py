from os import listdir
from os.path import exists
from lib.utils.misc.pathresolver import acrp

origpath = acrp('data/ilsvrc12/one_per_class')
resize = 190
crop = 190

origdirnames = listdir(origpath)
origdirnames.sort()

f = open(acrp('data/ilsvrc12/test_vis.txt'), 'w')

fallfiles = open(acrp('data/ilsvrc12/train.txt'))
allfiles = fallfiles.readlines()
fallfiles.close()

for file in origdirnames:
    filepath = origpath + '/' + file

    if not exists(filepath):
        print filepath + " doesn't exist, moving on..."
        continue

    print 'Searching for {0}...'.format(filepath)
    for p in allfiles:
        sp = p.split()
        fp = sp[0].split('/')[1]
        label = sp[1]
        if fp == file:
            print 'Found {0}'.format(fp)
            f.write('{0} {1}\n'.format(filepath, label))
            break

f.close()

print "Done!"
