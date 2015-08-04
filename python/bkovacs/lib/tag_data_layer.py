# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a network with multiple tags.
Tags can mean attributes or normal class labels.

TagDataLayer implements a Caffe Python layer.
"""

import json
import random

import numpy as np
import numpy.random as npr

import caffe
from minibatch import get_tag_minibatch
from utils.misc.progressbaraux import progress_bar


class TagDataLayer(caffe.Layer):
    """Image data layer used for training which can handle multiple tags the dataset."""

    def __init__(self, layer_param):
        super(TagDataLayer, self).__init__(layer_param)
        # Set default values
        self._random_seed = 0

    def _shuffle_db_inds(self):
        """Randomly permute the training set."""
        self._perm = npr.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the db indices for the next minibatch."""
        # Shuffle again, if we reached the end of the database or we just
        # started
        if self._cur == 0 or self._cur + self._ims_per_batch >= len(self._db):
            self._shuffle_db_inds()

        db_inds = self._perm[self._cur:self._cur + self._ims_per_batch]
        self._cur += self._ims_per_batch

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return get_tag_minibatch(
            minibatch_db,
            self._is_training,
            self._tag_names,
            self._num_classes,
            self._transformer,
            self._input_name,
            self._image_dims,
            self._crop_dims,
        )

    def _load_db(self):
        """Set the db to be used by this layer during training."""

        print 'Loading source file...'
        db = []
        with open(self._source) as f:
            for l in progress_bar(f.readlines()):
                json_data = json.loads(l)
                db.append(json_data)
        print 'Done.'

        self._db = db

    def _setup_from_params(self):
        # parse the layer parameter string, which must be valid JSON
        print 'Parsing json:', self.param_str_
        layer_params = json.loads(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._source = layer_params['source']
        self._ims_per_batch = layer_params['batch_size']
        # TODO: Use these parameters
        #self._shuffle = layer_params['shuffle']

        self._image_dims = (
            layer_params['new_height'],
            layer_params['new_width'],
        )
        self._crop_dims = (
            layer_params['crop_height'],
            layer_params['crop_width'],
        )
        self._mean = np.array(layer_params['mean'])

    def _setup_transformer(self):
        raw_scale = 255
        input_scale = 1
        channel_swap = (2, 1, 0)

        self._input_name = 'data'
        self._transformer = caffe.io.Transformer(
            {self._input_name: (self._ims_per_batch, 3) + self._crop_dims}
        )
        self._transformer.set_transpose(self._input_name, (2, 0, 1))
        if raw_scale is not None:
            self._transformer.set_raw_scale(self._input_name, raw_scale)
        if self._mean is not None:
            self._transformer.set_mean(self._input_name, self._mean)
        if input_scale is not None:
            self._transformer.set_input_scale(self._input_name, input_scale)
        if channel_swap is not None:
            self._transformer.set_channel_swap(self._input_name, channel_swap)

    def set_random_seed(self, random_seed):
        """Sets random seed, so we can have reproductible results."""
        self._random_seed = random_seed

    def set_is_training(self, is_training):
        """Sets is_training attribute so we know which phase we are in."""
        self._is_training = is_training

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        if hasattr(self, '_random_seed'):
            npr.seed(self._random_seed)
            random.seed(self._random_seed)

        # Getting values which were set by the Caffe Python wrapper
        self._is_training = self.phase_ == 'TRAIN'
        name_list = [name for name in self.top_names_]
        self._input_name = name_list[0]
        self._tag_names = name_list[1:]
        self._name_to_top_map = {name: i for i, name in enumerate(name_list)}

        self._cur = 0
        self._setup_from_params()
        self._setup_transformer()

        # Load db from textfile
        self._load_db()

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._ims_per_batch, 3, *self._crop_dims)

        # label blob: N categorical labels in [0, ..., K] for K classes
        for i, nc in enumerate(self._num_classes):
            top[i + 1].reshape(self._ims_per_batch, nc)

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        blobs = self._get_next_minibatch()

        # TODO: Performance evaluation of this layer, see why it is so slow...
        for blob_name, blob in blobs.iteritems():
            top_ind = self._name_to_top_map[blob_name]
            # Reshape net's input blobs
            top[top_ind].reshape(*(blob.shape))
            # Copy data into net's input blobs
            top[top_ind].data[...] = blob.astype(np.float32, copy=False)

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        """Reshaping happens during the call to forward."""
        pass
