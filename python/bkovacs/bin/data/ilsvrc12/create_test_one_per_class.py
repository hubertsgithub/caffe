from os import listdir
from os.path import exists
from lib.utils.misc.pathresolver import acrp

origpath = acrp('data/ilsvrc12/one_per_class')
resize = 190
crop = 190

origdirnames = listdir(origpath)
origdirnames.sort()

f = open(acrp('data/ilsvrc12/test_vis.txt'), 'w')
fwords = open(acrp('data/ilsvrc12/words_classnumbers.txt'), 'w')

fallfiles = open(acrp('data/ilsvrc12/train.txt'))
allfiles = fallfiles.readlines()
fallfiles.close()

fsynsetwords = open(acrp('data/ilsvrc12/synset_words.txt'))
synsetwords = fsynsetwords.readlines()
fsynsetwords.close()

for file in origdirnames:
    filepath = origpath + '/' + file

    if not exists(filepath):
        print filepath + " doesn't exist, moving on..."
        continue

    print 'Searching for {0}...'.format(filepath)
    for p in allfiles:
        sp = p.split()
        fp = sp[0].split('/')[1]
        label = int(sp[1])
        if fp == file:
            print 'Found {0}'.format(fp)
            f.write('{0} {1}\n'.format(filepath, label))
            synsetword = synsetwords[label]
            ssw = synsetword.split(' ')
            fwords.write('{0} {1}\n'.format(label, ' '.join(ssw[1:])))
            break

f.close()
fwords.close()

print "Done!"
