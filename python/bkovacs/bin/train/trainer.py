#!/usr/bin/python

import os
import re
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from Queue import Queue

import numpy as np

from lib.utils.data import fileproc
from lib.utils.misc import plothelper

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
    elif keyword == 'redirect':
        if value != 'True' and value != 'False':
            raise ValueError('Invalid redirect value: {0}'.format(value))
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
    if len(argv) < 4:
        raise ValueError('Too few arguments!')
    argv = argv[1:]

    options = dict(process_arg(argstr) for argstr in argv)
    print options

    # Default values
    mandatory_param_check(options, 'root')
    mandatory_param_check(options, 'modelname')
    optional_param_default(options, 'redirect', 'True')
    optional_param_default(options, 'platform', 'GPU')

    return options


def line_processor(identifier, line, itnum, train_losses, test_losses, test_accuracies):
    float_pattern = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?'
    itnum_pattern = 'Iteration (\\d+)'
    train_loss_pattern = 'Train net output #\\d: loss = ({0})'.format(float_pattern)
    test_loss_pattern = 'Test net output #\\d: loss = ({0})'.format(float_pattern)
    test_accuracy_pattern = 'Test net output #\\d: accuracy = ({0})'.format(float_pattern)

    print '{0}: {1}'.format(identifier, line.strip('\n'))

    match = re.search(itnum_pattern, line)
    if match:
        itnum = match.groups()[0]
        itnum = int(itnum)

    match = re.search(train_loss_pattern, line)
    if match:
        train_loss = match.groups()[0]
        train_loss = float(train_loss)
        train_losses[itnum] = train_loss

    match = re.search(test_loss_pattern, line)
    if match:
        test_loss = match.groups()[0]
        test_loss = float(test_loss)
        test_losses[itnum] = test_loss

    match = re.search(test_accuracy_pattern, line)
    if match:
        test_accuracy = match.groups()[0]
        test_accuracy = float(test_accuracy)
        test_accuracies[itnum] = test_accuracy

    return itnum


def create_figure_data(data_dict):
    ret = np.empty((len(data_dict), 2))
    for idx, (itnum, val) in enumerate(data_dict.iteritems()):
        ret[idx, 0] = itnum
        ret[idx, 1] = val

    return ret


def create_csv_data(data_dict):
    ret = []
    for itnum, val in data_dict.iteritems():
        ret.append('{0}, {1}'.format(itnum, val))

    return ret


def output_processor(stdout_queue, stderr_queue, update_interval, figure_filepath_root):
    # key: itnum, value: loss value
    itnum = 0
    train_losses = OrderedDict()
    test_losses = OrderedDict()
    test_accuracies = OrderedDict()

    # Check the queues if we received some output (until there is nothing more to get).
    while not stdout_reader.eof() or not stderr_reader.eof():
        new_data = False
        # Show what we received from standard output.
        while not stdout_queue.empty():
            line = stdout_queue.get()
            itnum = line_processor('STDOUT', line, itnum, train_losses, test_losses, test_accuracies)
            new_data = True

        # Show what we received from standard error.
        while not stderr_queue.empty():
            line = stderr_queue.get()
            itnum = line_processor('STERR', line, itnum, train_losses, test_losses, test_accuracies)
            new_data = True

        if train_losses and test_losses and itnum != 0 and new_data:
            # Save new plots
            train_figure_arr = create_figure_data(train_losses)
            test_figure_arr = create_figure_data(test_losses)
            max_loss = max(np.max(train_figure_arr[:, 1]), np.max(test_figure_arr[:, 1]))
            loss_figure_filepath = '{0}-loss.png'.format(figure_filepath_root)
            plothelper.plot_and_save_2D_arrays(loss_figure_filepath, [train_figure_arr, test_figure_arr], xlabel='Iteration number', xinterval=[0, itnum], ylabel='Loss', yinterval=[0, max_loss*1.1], line_names=['Train loss', 'Test loss'])

            train_loss_data_filepath = '{0}-train-loss.txt'.format(figure_filepath_root)
            fileproc.fwritelines(train_loss_data_filepath, create_csv_data(train_losses))
            test_loss_data_filepath = '{0}-test-loss.txt'.format(figure_filepath_root)
            fileproc.fwritelines(test_loss_data_filepath, create_csv_data(test_losses))

        if test_accuracies and itnum != 0 and new_data:
            # Save new plots
            test_acc_figure_filepath = '{0}-test-accuracy.png'.format(figure_filepath_root)
            test_acc_figure_arr = create_figure_data(test_accuracies)
            plothelper.plot_and_save_2D_array(test_acc_figure_filepath, test_acc_figure_arr, xlabel='Iteration number', xinterval=[0, itnum], ylabel='Accuracy', yinterval=[0, 1])

            test_accuracy_data_filepath = '{0}-test-accuracy.txt'.format(figure_filepath_root)
            fileproc.fwritelines(test_accuracy_data_filepath, create_csv_data(test_accuracy_data_filepath))

        # Sleep a bit before asking the readers again.
        time.sleep(update_interval)


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
    if options['redirect'] == 'True':
        proc = subprocess.Popen(commandtxt, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

        # Launch the asynchronous readers of the process' stdout and stderr.
        stdout_queue = Queue()
        stdout_reader = fileproc.AsynchronousFileReader(proc.stdout, stdout_queue)
        stdout_reader.start()
        stderr_queue = Queue()
        stderr_reader = fileproc.AsynchronousFileReader(proc.stderr, stderr_queue)
        stderr_reader.start()

        output_processor(stdout_queue, stderr_queue, 1, os.path.join(options['root'], options['modelname']))

        # Let's be tidy and join the threads we've started.
        stdout_reader.join()
        stderr_reader.join()

        # Close subprocess' file descriptors.
        proc.stdout.close()
        proc.stderr.close()
    else:
        subprocess.call(commandtxt)
