# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a network with balanced training set.

RoIDataLayer implements a Caffe Python layer.
"""

import json
import random

import numpy as np
import numpy.random as npr

import caffe
from minibatch import get_minibatch


class BalancedImageDataLayer(caffe.Layer):
    """Image data layer used for training which can balance the dataset."""

    def __init__(self):
        super(BalancedImageDataLayer, self).__init__(self)
        # Set default values
        self._random_seed = 0
        self._is_training = True

    def _shuffle_db_inds(self):
        """Randomly permute the training set."""
        if self._balance:
            return

        self._perm = npr.permutation(np.arange(len(self._db)))
        self._cur = 0

    def _get_next_minibatch_inds(self):
        """Return the db indices for the next minibatch."""
        if self._balance:
            # Choose random label with uniform probability
            labels = npr.randint(
                0, self._num_classes, size=(self._ims_per_batch)
            )
            # Choose random image for the specified label
            db_inds = [
                npr.choice(self._db_by_label[l])
                for l in labels
            ]
        else:
            # Shuffle again, if we reached the end of the database or we just
            # started
            if self._cur == 0 or self._cur + self._ims_per_batch >= len(self._db):
                self._shuffle_db_inds()

            db_inds = self._perm[self._cur:self._cur + self._ims_per_batch]
            self._cur += self._ims_per_batch

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        Blobs will be computed in a separate process and made available through
        self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return get_minibatch(
            self._is_training,
            minibatch_db,
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
        db_by_label = [[] for i in range(self._num_classes)]
        with open(self._source) as f:
            for i, l in enumerate(f.readlines()):
                img_path, label = l.split(' ')
                label = int(label)
                db.append({
                    'image': img_path,
                    'label': label,
                })
                db_by_label[label].append(i)
        print 'Done.'

        self._db = db
        self._db_by_label = db_by_label

    def _setup_from_params(self):
        # parse the layer parameter string, which must be valid JSON
        print 'Parsing json:', self.param_str_
        layer_params = json.loads(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._source = layer_params['source']
        self._ims_per_batch = layer_params['batch_size']
        # TODO: Use these parameters
        #self._shuffle = layer_params['shuffle']
        self._balance = layer_params['balance']

        self._image_dims = (
            layer_params['new_height'],
            layer_params['new_width'],
        )
        self._crop_dims = (
            layer_params['crop_height'],
            layer_params['crop_width'],
        )

        raw_scale = 255
        mean = np.array(layer_params['mean'])
        input_scale = 1
        channel_swap = (2, 1, 0)

        self._input_name = 'data'
        self._transformer = caffe.io.Transformer(
            {self._input_name: (self._ims_per_batch, 3) + self._crop_dims}
        )
        self._transformer.set_transpose(self._input_name, (2, 0, 1))
        if raw_scale is not None:
            self._transformer.set_raw_scale(self._input_name, raw_scale)
        if mean is not None:
            self._transformer.set_mean(self._input_name, mean)
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

        self._cur = 0
        self._setup_from_params()

        # Load db from textfile
        self._load_db()

        self._name_to_top_map = {
            'data': 0,
            'label': 1,
        }

        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._ims_per_batch, 3, *self._crop_dims)

        # label blob: N categorical labels in [0, ..., K] for K classes
        top[1].reshape(self._ims_per_batch)

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
