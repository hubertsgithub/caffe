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
    argv = argv[1:]
    options = dict(process_arg(argstr) for argstr in argv)

    # Default values
    mandatory_param_check(options, 'root')
    mandatory_param_check(options, 'modelname')
    optional_param_default(options, 'redirect', 'True')
    optional_param_default(options, 'platform', 'GPU')

    print options

    return options


def line_processor(identifier, line, itnum, outputs, output_names):
    float_pattern = '[-+]?[0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?'
    itnum_pattern = 'Iteration (\\d+)'
    train_output_pattern = 'Train net output #(\\d+): (\S+) = ({0})'.format(float_pattern)
    test_output_pattern = 'Test net output #(\\d+): (\S+) = ({0})'.format(float_pattern)
    patterns = [train_output_pattern, test_output_pattern]

    print '{0}: {1}'.format(identifier, line.strip('\n'))

    match = re.search(itnum_pattern, line)
    if match:
        itnum = match.groups()[0]
        itnum = int(itnum)

    for i in range(2):
        match = re.search(patterns[i], line)
        if match:
            output_num, output_name, output = match.groups()
            output = float(output)
            output_num = int(output_num)

            output_names[i][output_num] = output_name
            if output_num not in outputs[i]:
                outputs[i][output_num] = OrderedDict()
            outputs[i][output_num][itnum]= output

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
    itnum = 0
    # index 0 is train, 1 is test
    # {key: itnum, value: {key: output_num value: output_value}}
    outputs = [OrderedDict(), OrderedDict()]
    # {key: output_num, value: output_name}
    output_names = [OrderedDict(), OrderedDict()]

    disp_config = OrderedDict()
    # loss
    disp_config['loss'] = {'file_name': 'loss', 'yaxis_name': 'Loss', 'yinterval_policy': 'max'}
    # accuracy
    disp_config['accuracy'] = {'file_name': 'accuracy', 'yaxis_name': 'Accuracy', 'yinterval_policy': 'fix', 'yinterval_value': [0, 1]}
    # other
    disp_config['other'] = {'file_name': 'output', 'yaxis_name': 'Output'}

    # Check the queues if we received some output (until there is nothing more to get).
    while not stdout_reader.eof() or not stderr_reader.eof():
        new_data = False
        # Show what we received from standard output.
        while not stdout_queue.empty():
            line = stdout_queue.get()
            itnum = line_processor('STDOUT', line, itnum, outputs, output_names)
            new_data = True

        # Show what we received from standard error.
        while not stderr_queue.empty():
            line = stderr_queue.get()
            itnum = line_processor('STDERR', line, itnum, outputs, output_names)
            new_data = True

        if itnum != 0 and new_data:
            # Partition outputs into 'loss', 'accuracy', 'other'
            # contains index pairs: 0/1 (training/test) and output_num
            indices = {}
            for disp_type in disp_config:
                indices[disp_type] = []

            for i in range(2):
                for output_num, output_name in output_names[i].iteritems():
                    found_type = False
                    for disp_type in disp_config:
                        if disp_type in output_name:
                            indices[disp_type].append([i, output_num])
                            found_type = True

                    if not found_type:
                        indices['other'].append([i, output_num])

            for disp_type, dc in disp_config.iteritems():
                figure_arrs = []
                line_names = []
                line_template_str = ['Train {0}', 'Test {0}']
                csv_file_template_str = ['{0}-train-{1}.txt', '{0}-test-{1}.txt']

                for i, output_num in indices[disp_type]:
                    op = outputs[i][output_num]
                    on = output_names[i][output_num]

                    figure_arrs.append(create_figure_data(op))
                    line_names.append(line_template_str[i].format(on))

                    data_filepath = csv_file_template_str[i].format(figure_filepath_root, on)
                    fileproc.fwritelines(data_filepath, create_csv_data(op))

                if not figure_arrs:
                    continue

                # Choose different settings depending on the config
                if dc['yinterval_policy'] == 'max':
                    ymax = np.max([np.max(fa[:, 1]) for fa in figure_arrs])
                    yinterval = [0, ymax*1.1]
                elif dc['yinterval_policy'] == 'fix':
                    yinterval = dc['yinterval_value']
                else:
                    yinterval = None

                figure_filepath = '{0}-{1}.png'.format(figure_filepath_root, dc['file_name'])
                plothelper.plot_and_save_2D_arrays(figure_filepath, figure_arrs, xlabel='Iteration number', xinterval=[0, itnum], ylabel=dc['yaxis_name'], yinterval=yinterval, line_names=line_names)

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
