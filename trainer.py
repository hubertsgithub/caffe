#!/usr/bin/python

import sys
import os
import subprocess

if len(sys.argv) != 3 and len(sys.argv) != 4:
    print 'Correct usage: python trainer.py <root> <modelname> <platform=(CPU,GPU)?>'
    sys.exit()

root = sys.argv[1]
modelname = sys.argv[2]
platform = 'GPU'
if len(sys.argv) == 4:
    if sys.argv[3] == 'CPU' or sys.argv[3] == 'GPU':
        platform = sys.argv[3]
    else:
        print 'Correct usage: /trainer.py <modelname> <platform=(CPU,GPU)?>'
        sys.exit()

print 'Running training for model {0}...'.format(modelname)

samplesolverfilename = 'solver.prototxt'
trainfilename = 'train_val_{0}.prototxt'.format(modelname)
solverfilename = 'solver_{0}.prototxt'.format(modelname)

if not os.path.exists(os.path.join(root, trainfilename)):
    print 'Traning file {0} doesn\'t exist! Please provide a valid model name!'.format(trainfilename)
    sys.exit()

sample_to_use = ''
# if the solver file exists, just use it
if os.path.exists(os.path.join(root, solverfilename)):
    sample_to_use = solverfilename
else:
    sample_to_use = samplesolverfilename

# copy the sample solver file and modify it
fin = open(os.path.join(root, sample_to_use), 'r')
lines = fin.readlines()
fin.close()

fout = open(os.path.join(root, solverfilename), 'w')
for l in lines:
    newl = l.replace('train_val.prototxt', trainfilename)
    newl = newl.replace('caffenet_train', 'caffenet_train_{0}'.format(modelname))
    newl = newl.replace('solver_mode: GPU', 'solver_mode: {0}'.format(platform))
    fout.write(newl)

fout.close()

commandtxt = ['./build/tools/caffe', 'train', '--solver={0}'.format(os.path.join(root, solverfilename))]
print 'Running command \'{0}\'...'.format(' '.join(commandtxt))
subprocess.call(commandtxt)

