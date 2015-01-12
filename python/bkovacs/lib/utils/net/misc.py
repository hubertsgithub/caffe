import sys

import numpy as np

from lib.utils.misc.pathresolver import acrp

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe


def init_net(model_file, pretrained_weights, mean, input_names, channel_swap=(2, 1, 0), raw_scale=255, input_scale=1):
    net = caffe.Net(model_file, pretrained_weights)
    net.set_phase_test()
    net.set_mode_cpu()
    for input_name in input_names:
        net.set_channel_swap(input_name, channel_swap)
        net.set_raw_scale(input_name, raw_scale)
        net.set_input_scale(input_name, input_scale)

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


