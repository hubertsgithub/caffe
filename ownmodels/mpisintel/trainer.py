#!/usr/bin/python

import sys
import os
import subprocess

if len(sys.argv) != 2:
    print 'Correct usage: /trainer.py <modelname>'
    sys.exit()

root = 'ownmodels/mpisintel/'
modelname = sys.argv[1]
print 'Running training for model {0}...'.format(modelname)

samplesolverfilename = 'solver.prototxt'
trainfilename = 'train_val_{0}.prototxt'.format(modelname)
solverfilename = 'solver_{0}.prototxt'.format(modelname)

if not os.path.exists(root + trainfilename):
    print 'Traning file {0} doesn\'t exist! Please provide a valid model name!'.format(trainfilename)
    sys.exit()

# copy the sample solver file and modify it
fin = open(root + samplesolverfilename, 'r')
lines = fin.readlines()
fin.close()

fout = open(root + solverfilename, 'w')
for l in lines:
    newl = l.replace('train_val.prototxt', trainfilename)
    newl = newl.replace('caffenet_train', 'caffenet_train_{0}'.format(modelname))
    fout.write(newl)

fout.close()

commandtxt = ['./build/tools/caffe', 'train', '--solver={0}{1}'.format(root, solverfilename)]
print 'Running command \'{0}\'...'.format(' '.join(commandtxt))
subprocess.call(commandtxt)

