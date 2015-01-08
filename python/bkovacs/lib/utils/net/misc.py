import sys

import numpy as np

from lib.utils.misc.pathresolver import acrp

# Make sure that caffe is on the python path:
sys.path.append(acrp('python'))
import caffe


def init_net(model_file, pretrained_weights, mean_file, input_name):
    net = caffe.Net(model_file, pretrained_weights)
    net.set_phase_test()
    net.set_mode_cpu()
    net.set_channel_swap(input_name, (2, 1, 0))
    net.set_raw_scale(input_name, 255)

    blob = caffe.proto.caffe_pb2.BlobProto()
    data = open(mean_file, 'rb').read()
    blob.ParseFromString(data)
    meanarr = caffe.io.blobproto_to_array(blob)
    # Remove the first dimension (batch), which is 1 anyway
    meanarr = np.squeeze(meanarr, axis=0)

    net.set_mean(input_name, meanarr)

    return net


