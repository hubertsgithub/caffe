# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""The abstract parent of all python data layers. PythonDataLayer implements a Caffe Python layer.
"""

import json
import os
import random
import signal
from Queue import Full, Queue
from threading import Thread

import numpy as np
import numpy.random as npr

import caffe


class PythonDataLayer(caffe.Layer):
    """Image data layer used for training which can handle multiple tags the dataset."""

    def __init__(self, layer_param):
        super(PythonDataLayer, self).__init__(layer_param)
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
        raise NotImplementedError('Please override this method in the child class')

    def _load_db(self):
        """Set the db to be used by this layer during training."""
        raise NotImplementedError('Please override this method in the child class')

    def _setup_from_params(self):
        # parse the layer parameter string, which must be valid JSON
        print 'Parsing json:', self.param_str_
        layer_params = json.loads(self.param_str_)
        self._num_classes = layer_params['num_classes']
        self._source = layer_params['source']
        self._ims_per_batch = layer_params['batch_size']
        if 'freqs' in layer_params:
            # This attribute should point to a JSON file which contains a
            # dictionary which has a one dimensional array for each label/tag
            # type
            self._freqs = json.load(open(layer_params['freqs']))
        else:
            self._freqs = None

        self._setup_extra_from_params(layer_params)
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

    def _setup_extra_from_params(self, layer_params):
        '''This method can be overridden to extract more parameters from the
        param_str'''
        pass

    def _setup_transformer(self):
        raw_scale = 255
        input_scale = 1
        channel_swap = (2, 1, 0)

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

    def _setup_blobfetcher(self):
        self._blob_queue = Queue(10)
        self._prefetch_thread = BlobFetcher(
            self._blob_queue, self
        )
        self._prefetch_thread.start()

        def signal_handler(signal, frame):
            print 'Stopping BlobFetcher...'
            self._prefetch_thread.stop()
            print 'Joining BlobFetcher thread...'
            self._prefetch_thread.join()
            print 'Exiting...'
            # "Hard" exit, because sys.exit won't exit the C++, because the
            # exception gets intercepted...
            os._exit(0)
        signal.signal(signal.SIGINT, signal_handler)

    def setup(self, bottom, top):
        """Setup the DataLayer."""
        if hasattr(self, '_random_seed'):
            npr.seed(self._random_seed)
            random.seed(self._random_seed)

        self._setup_from_params()

        # Getting values which were set by the Caffe Python wrapper
        self._is_training = self.phase_ == 'TRAIN'
        name_list = [name for name in self.top_names_]
        self._input_name = name_list[0]
        nc_num = len(self._num_classes)
        self._tag_names = name_list[1:nc_num+1]
        # If frequencies were defined, they should be the last top blobs
        if self._freqs:
            self._freq_names = name_list[nc_num+1:2*nc_num+1]

        self._name_to_top_map = {name: i for i, name in enumerate(name_list)}

        self._cur = 0
        self._setup_transformer()

        # Load db from textfile
        self._load_db()
        self._reshape_tops(top)
        #self._setup_blobfetcher()

    def _reshape_tops(self, top):
        """Give initial shape to the top blobs according to the parsed layer
        parameters"""
        raise NotImplementedError('Please override this method in the child class')

    def forward(self, bottom, top):
        """Get blobs and copy them into this layer's top blob vector."""
        #blobs = self._blob_queue.get()
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


class BlobFetcher(Thread):
    """Class for prefetching blobs on a separate thread."""
    def __init__(self, queue, layer):
        super(BlobFetcher, self).__init__()
        self._queue = queue
        self._layer = layer
        self._stopping = False

    def run(self):
        print 'BlobFetcher started'
        while not self._stopping:
            print 'Getting minibatch in BlobFetcher...'
            blobs = self._layer._get_next_minibatch()
            while not self._stopping:
                # If the queue is full, retry until we can put the blobs in the
                # queue
                try:
                    self._queue.put(blobs, timeout=1)
                    break
                except Full:
                    pass
        print 'BlobFetcher exited'

    def stop(self):
        self._stopping = True
