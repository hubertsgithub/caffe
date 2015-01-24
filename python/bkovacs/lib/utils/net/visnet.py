import os
import random

from PIL import Image

from lib.utils.misc import plothelper, images2gif
from lib.utils.net.misc import init_net


def vis_net(filepath_root, postfix, model_file, pretrained_weights, ratio=1):
    net = init_net(model_file, pretrained_weights)
    image_files = {}

    for layer_name, (weights, biases) in net.params.iteritems():
        # dimensions: channels, num, height, width
        filepath = '{0}-{1}-{2}.png'.format(filepath_root, layer_name, postfix)

        # get a sample from the channels based on ratio
        output_channels, input_channels, height, width = weights.data.shape
        random.seed(42)
        indices = random.sample(xrange(output_channels), int(output_channels*ratio))

        # sample channels using the indices
        plothelper.save_vis_square(filepath, weights.data[indices])
        image_files[layer_name] = filepath

    return image_files


def create_gif(filepath_root, image_files, duration=0.5):
    image_files_by_layer = {}
    for imfs in image_files:
        for layer_name, image_file in imfs.iteritems():
            if layer_name not in image_files_by_layer:
                image_files_by_layer[layer_name] = []

            image_files_by_layer[layer_name].append(image_file)

    for layer_name, imfs in image_files_by_layer.iteritems():
        filepath = '{0}-{1}.gif'.format(filepath_root, layer_name)
        imgs = [Image.open(imf) for imf in imfs]

        images2gif.writeGif(filepath, imgs, duration=duration)


