import sys

import numpy as np

from lib.utils.misc.pathresolver import acrp

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe


def init_net(model_file, pretrained_weights, mean, input_config):
    net = caffe.Net(model_file, pretrained_weights)
    net.set_phase_test()
    net.set_mode_cpu()
    for input_name, config in input_config.iteritems():
        if 'channel_swap' in config:
            net.set_channel_swap(input_name, config['channel_swap'])
        if 'raw_scale' in config:
            net.set_raw_scale(input_name, config['raw_scale'])
        if 'input_scale' in config:
            net.set_input_scale(input_name, config['input_scale'])

        if isinstance(mean, basestring):
            blob = caffe.proto.caffe_pb2.BlobProto()
            data = open(mean_file, 'rb').read()
            blob.ParseFromString(data)
            meanarr = caffe.io.blobproto_to_array(blob)
            # Remove the first dimension (batch), which is 1 anyway
            meanarr = np.squeeze(meanarr, axis=0)

            net.set_mean(input_name, meanarr, mode='elementwise')
        elif isinstance(mean, int) or isinstance(mean, float):
            meanarr = np.empty((1, 3))
            meanarr.fill(mean)
            net.set_mean(input_name, meanarr, mode='channel')

    return net


