#!/usr/bin/python

import os
import re
import subprocess
import sys
import time
from collections import OrderedDict
from datetime import datetime
from pipes import quote
from Queue import Queue

from caffe.proto import caffe_pb2
from lib.utils.data import caffefileproc, common, fileproc
from lib.utils.misc.pathresolver import acrp

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))


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
            outputs[i][output_num][itnum] = output

    return itnum


def output_processor(stdout_queue, stderr_queue,
                     stdout_reader, stderr_reader,
                     update_interval,
                     caffe_cnn_trrun_id, data_callback):
    itnum = 0
    # index 0 is train, 1 is test
    # {key: output_num, value: {key: it_num value: output_value}}
    outputs = [OrderedDict(), OrderedDict()]
    # {key: output_num, value: output_name}
    output_names = [OrderedDict(), OrderedDict()]

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
            # Call the callback function to notify the views about new data
            data_callback(caffe_cnn_trrun_id, outputs, output_names, itnum)

        # Sleep a bit before asking the readers again.
        time.sleep(update_interval)


def extract_batchsize_testsetsize(model_file_content):
    # Note that we handle only MULTI_IMAGE_DATA layer and some other layers (see below), we require that include.phase be TEST
    model_params = caffefileproc.parse_model_definition_file_content(model_file_content)
    batch_size = None
    testset_size = None
    for layer in model_params.layers:
        if layer.type in [caffe_pb2.LayerParameter.MULTI_IMAGE_DATA, caffe_pb2.LayerParameter.MULTI_IMAGE_PATCH_DATA, caffe_pb2.LayerParameter.IMAGE_DATA]:
            if layer.include[0].phase == caffe_pb2.TEST:
                batch_size = layer.image_data_param.batch_size
                source = layer.image_data_param.source
                # Note that the source should be relative to the caffe root path!
                testset_size = len(fileproc.freadlines(acrp(source)))
                break

    return batch_size, testset_size


def start_training(model_name, model_file_content, solver_file_content,
                   options, caffe_cnn_trrun_id, data_callback):
    print 'Running training for model {}...'.format(model_name)
    print 'with options: {}'.format(options)

    rand_name = str(datetime.now())
    root_path = acrp(os.path.join('training_runs', '-'.join([rand_name, model_name])))

    common.ensuredir(root_path)
    trainfilename = 'train_val.prototxt'
    solverfilename = 'solver.prototxt'
    trainfile_path = os.path.join(root_path, trainfilename)
    solverfile_path = os.path.join(root_path, solverfilename)

    # Save the model_file_content to a file, so Caffe can read it
    with open(trainfile_path, 'w') as f:
        f.write(model_file_content)

    # copy the sample solver file and modify it
    solver_params = caffefileproc.parse_solver_file_content(solver_file_content)

    # modify solver params according to the command line parameters
    solver_params.net = trainfile_path
    if 'base_lr' in options and options['base_lr'] is not None:
        solver_params.base_lr = options['base_lr']

    # Switch on debug_info to facilitate debugging
    if 'debug_info' in options and options['debug_info'] is not None:
        solver_params.debug_info = options['debug_info']

    snapshot_path = os.path.join(root_path, 'snapshots')
    common.ensuredir(snapshot_path)
    solver_params.snapshot_prefix = os.path.join(
        snapshot_path,
        'train_{}-base_lr{}'.format(model_name, solver_params.base_lr)
    )
    # The default is GPU
    if 'cpu' in options and options['cpu']:
        solver_params.solver_mode = caffe_pb2.SolverParameter.CPU
    else:
        solver_params.solver_mode = caffe_pb2.SolverParameter.GPU

    # compute the proper test_iter
    batch_size, testset_size = extract_batchsize_testsetsize(model_file_content)

    if batch_size and testset_size:
        print 'Extracted batch_size ({0}) and testset_size ({1})'.format(
            batch_size, testset_size)
        # Note the solver file should have exactly one test_iter
        solver_params.test_iter[0] = int(testset_size/batch_size)
    else:
        print 'WARNING: Couldn\'t find the batch_size or the source file ' + \
            'containing the testset, please set the test_iter to ' + \
            'testset_size / batch_size!'

    caffefileproc.save_protobuf_file(solverfile_path, solver_params)

    commandtxt = ['./build/tools/caffe', 'train', '--solver={0}'.format(quote(solverfile_path))]
    if 'weights' in options and options['weights'] is not None:
        # If we gave a solverstate file, we have to call with '--snapshot',
        # otherwise we use '--weights'
        _, ext = os.path.splitext(options['weights'])
        if ext == '.solverstate':
            prefix = 'snapshot'
        else:
            prefix = 'weights'

        commandtxt.append('--{}={}'.format(prefix, quote(acrp(options['weights']))))

    print 'Running command \'{0}\'...'.format(' '.join(commandtxt))
    # Set the caffe root path as the working directory of the command
    proc = subprocess.Popen(
        commandtxt,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=acrp('')
    )

    # Launch the asynchronous readers of the process' stdout and stderr.
    stdout_queue = Queue()
    stdout_reader = fileproc.AsynchronousFileReader(proc.stdout, stdout_queue)
    stdout_reader.start()
    stderr_queue = Queue()
    stderr_reader = fileproc.AsynchronousFileReader(proc.stderr, stderr_queue)
    stderr_reader.start()

    output_processor(
        stdout_queue=stdout_queue,
        stderr_queue=stderr_queue,
        stdout_reader=stdout_reader,
        stderr_reader=stderr_reader,
        update_interval=1,
        caffe_cnn_trrun_id=caffe_cnn_trrun_id,
        data_callback=data_callback,
    )

    # Let's be tidy and join the threads we've started.
    stdout_reader.join()
    stderr_reader.join()

    # Close subprocess' file descriptors.
    proc.stdout.close()
    proc.stderr.close()
