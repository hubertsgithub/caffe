import os
import random

from PIL import Image

from lib.utils.misc import plothelper, images2gif
from lib.utils.net.misc import init_net


def vis_net(filepath_root, postfix, model_file, pretrained_weights, ratio=1):
    net = init_net(model_file, pretrained_weights)
    analytic_images = {}

    for layer_name, (weights, biases) in net.params.iteritems():
        # dimensions: channels, num, height, width
        # get a sample from the channels based on ratio
        output_channels, input_channels, height, width = weights.data.shape
        random.seed(42)
        indices = random.sample(xrange(output_channels), int(output_channels*ratio))

        # sample channels using the indices
        analytic_images[layer_name] = plothelper.create_analytic_image(weights.data[indices])

    return analytic_images


def create_gif(filepath_root, analytic_images, duration=0.5):
    analytic_images_by_layer = {}
    for imgs in analytic_images:
        for layer_name, img in imgs.iteritems():
            if layer_name not in analytic_images_by_layer:
                analytic_images_by_layer[layer_name] = []

            analytic_images_by_layer[layer_name].append(img)

    for layer_name, imgs in analytic_images_by_layer.iteritems():
        filepath = '{0}-{1}.gif'.format(filepath_root, layer_name)

        images2gif.writeGif(filepath, imgs, duration=duration)


