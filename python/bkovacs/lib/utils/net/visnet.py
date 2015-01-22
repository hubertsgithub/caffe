import os
import random

from lib.utils.misc import plothelper
from lib.utils.net.misc import init_net


def vis_net(filepath_root, postfix, model_file, pretrained_weights, ratio=1):
    net = init_net(model_file, pretrained_weights)

    for layer_name, (weights, biases) in net.params.iteritems():
        # dimensions: channels, num, height, width
        filepath = '{0}-{1}-{2}.png'.format(filepath_root, layer_name, postfix)

        # get a sample from the channels based on ratio
        output_channels, input_channels, height, width = weights.data.shape
        indices = random.sample(xrange(output_channels), int(output_channels*ratio))

        # sample channels using the indices
        plothelper.save_vis_square(filepath, weights.data[indices])

