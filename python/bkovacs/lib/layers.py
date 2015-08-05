# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The data layer used during training to train a network with balanced training set.

RoIDataLayer implements a Caffe Python layer.
"""

import numpy.random as npr

from minibatch import get_minibatch
from python_data_layer import PythonDataLayer


class BalancedImageDataLayer(PythonDataLayer):
    """Python data layer used as abstract parent for all the data layers implemented in Python."""

    def _shuffle_db_inds(self):
        """Randomly permute the training set."""
        if self._balance:
            return

        super(BalancedImageDataLayer, self)._shuffle_db_inds()

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
            db_inds = super(BalancedImageDataLayer, self)._get_next_minibatch_inds()

        return db_inds

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.

        Blobs will be computed in a separate process and made available through
        self._blob_queue.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        return get_minibatch(
            minibatch_db,
            self._is_training,
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

    def _setup_extra_from_params(self, layer_params):
        self._balance = layer_params['balance']

    def _reshape_tops(self, top):
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._ims_per_batch, 3, *self._crop_dims)

        # label blob: N categorical labels in [0, ..., K] for K classes
        top[1].reshape(self._ims_per_batch)
