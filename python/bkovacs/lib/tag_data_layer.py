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

import numpy as np

from minibatch import get_tag_minibatch
from python_data_layer import PythonDataLayer
from utils.misc.progressbaraux import progress_bar


class TagDataLayer(PythonDataLayer):
    """Image data layer used for training which can handle multiple tags the dataset."""

    def _get_next_minibatch(self):
        """Return the blobs to be used for the next minibatch.
        """
        db_inds = self._get_next_minibatch_inds()
        minibatch_db = [self._db[i] for i in db_inds]
        blobs = get_tag_minibatch(
            minibatch_db,
            self._is_training,
            self._tag_names,
            self._num_classes,
            self._transformer,
            self._input_name,
            self._image_dims,
            self._crop_dims,
            self._make_one_tag_blob,
        )
        # If frequencies were defined, they should be the last top blobs
        if self._freqs:
            for fn in self._freq_names:
                blobs[fn] = np.array(self._freqs[fn])

        return blobs

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

    def _setup_extra_from_params(self, layer_params):
        '''This method can be overridden to extract more parameters from the
        param_str'''
        # This basically means that we fall back to the original "labels", i.e.
        # we have one tag / item.
        if 'make_one_tag_blob' not in layer_params:
            self._make_one_tag_blob = False
        else:
            self._make_one_tag_blob = layer_params['make_one_tag_blob']

    def _reshape_tops(self, top):
        # data blob: holds a batch of N images, each with 3 channels
        top[0].reshape(self._ims_per_batch, 3, *self._crop_dims)

        # label blob: N categorical labels in [0, ..., K] for K classes
        for i, nc in enumerate(self._num_classes):
            if self._make_one_tag_blob:
                top[i + 1].reshape(self._ims_per_batch, 1)
            else:
                top[i + 1].reshape(self._ims_per_batch, nc)

        # Blobs containing the label frequencies for balancing
        if self._freqs:
            for i, nc in enumerate(self._num_classes):
                top[i + 1 + len(self._num_classes)].reshape(nc)
