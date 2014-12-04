#!/usr/bin/python

import sys
import os
import subprocess

if len(sys.argv) != 2 and len(sys.argv) != 3:
    print 'Correct usage: /trainer.py <modelname> <platform=(CPU,GPU)?>'
    sys.exit()

root = 'ownmodels/mitintrinsic/'
modelname = sys.argv[1]
platform = 'GPU'
if len(sys.argv) == 3:
    if sys.argv[2] == 'CPU' or sys.argv[2] == 'GPU':
        platform = sys.argv[2]
    else:
        print 'Correct usage: /trainer.py <modelname> <platform=(CPU,GPU)?>'
        sys.exit()

print 'Running training for model {0}...'.format(modelname)

samplesolverfilename = 'solver.prototxt'
trainfilename = 'train_val_{0}.prototxt'.format(modelname)
solverfilename = 'solver_{0}.prototxt'.format(modelname)

if not os.path.exists(root + trainfilename):
    print 'Traning file {0} doesn\'t exist! Please provide a valid model name!'.format(trainfilename)
    sys.exit()

# if the solver file exists, just use it
if not os.path.exists(root + solverfilename):
    # copy the sample solver file and modify it
    fin = open(root + samplesolverfilename, 'r')
    lines = fin.readlines()
    fin.close()

    fout = open(root + solverfilename, 'w')
    for l in lines:
        newl = l.replace('train_val.prototxt', trainfilename)
        newl = newl.replace('caffenet_train', 'caffenet_train_{0}'.format(modelname))
        newl = newl.replace('solver_mode: GPU', 'solver_mode: {0}'.format(platform))
        fout.write(newl)

    fout.close()

commandtxt = ['./build/tools/caffe', 'train', '--solver={0}{1}'.format(root, solverfilename)]
print 'Running command \'{0}\'...'.format(' '.join(commandtxt))
subprocess.call(commandtxt)

