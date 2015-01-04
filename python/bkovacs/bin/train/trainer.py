#!/usr/bin/python

import sys
import os
import subprocess

CORRECTUSAGESTR = 'Correct usage: python trainer.py <root=rootpath> <modelname=the name of the model> <weights=pathtoweights?> <platform=(CPU,GPU)?>'


def print_and_exit():
    print CORRECTUSAGESTR
    sys.exit()


def process_arg(argstr):
    if argstr.find('=') == -1:
        raise ValueError('All parameters should have \'keyword=value\' format!')

    keyword, value = argstr.split('=')

    if keyword == 'root':
        if not os.path.exists(value):
            raise ValueError('The root path doesn\'t exist: {0}'.format(value))
    elif keyword == 'modelname':
        dummy = 0
    elif keyword == 'weights':
        if not os.path.exists(value):
            raise ValueError('The weight file doesn\'t exist: {0}'.format(value))
    elif keyword == 'platform':
        if value != 'CPU' and value != 'GPU':
            raise ValueError('Invalid platform value: {0}'.format(value))
    else:
        raise ValueError('Invalid keyword: {0}'.format(keyword))

    return keyword, value


def mandatory_param_check(options, paramname):
    if paramname not in options:
        raise ValueError('Param {0} is a mandatory parameter!'.format(paramname))


def optional_param_default(options, paramname, default):
    if paramname not in options:
        options[paramname] = default


def process_args(argv):
    if len(argv) < 3:
        raise ValueError('Too few arguments!')
    argv = argv[1:]

    options = dict(process_arg(argstr) for argstr in argv)
    print options

    # Default values
    mandatory_param_check(options, 'root')
    mandatory_param_check(options, 'modelname')
    optional_param_default(options, 'platform', 'GPU')

    return options


if __name__ == '__main__':
    options = process_args(sys.argv)
    print 'Running training for model {0}...'.format(options['modelname'])

    samplesolverfilename = 'solver.prototxt'
    trainfilename = 'train_val_{0}.prototxt'.format(options['modelname'])
    solverfilename = 'solver_{0}.prototxt'.format(options['modelname'])

    if not os.path.exists(os.path.join(options['root'], trainfilename)):
        print 'Traning file {0} doesn\'t exist! Please provide a valid model name!'.format(trainfilename)
        sys.exit()

    sample_to_use = ''
    # if the solver file exists, just use it
    if os.path.exists(os.path.join(options['root'], solverfilename)):
        sample_to_use = solverfilename
    else:
        sample_to_use = samplesolverfilename

    # copy the sample solver file and modify it
    fin = open(os.path.join(options['root'], sample_to_use), 'r')
    lines = fin.readlines()
    fin.close()

    fout = open(os.path.join(options['root'], solverfilename), 'w')
    for l in lines:
        newl = l.replace('train_val.prototxt', trainfilename)
        newl = newl.replace('caffenet_train\"', 'caffenet_train_{0}\"'.format(options['modelname']))
        newl = newl.replace('solver_mode: GPU', 'solver_mode: {0}'.format(options['platform']))
        fout.write(newl)

    fout.close()

    commandtxt = ['./build/tools/caffe', 'train', '--solver={0}'.format(os.path.join(options['root'], solverfilename))]
    if 'weights' in options:
        commandtxt.append('--weights=' + options['weights'])

    print 'Running command \'{0}\'...'.format(' '.join(commandtxt))
    subprocess.call(commandtxt)

