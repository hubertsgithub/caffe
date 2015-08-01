# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import numpy as np
import numpy.random as npr

import caffe


def get_minibatch(db, num_classes, transformer, input_name, image_dims, crop_dims):
    """Given a db, construct a minibatch sampled from it."""
    num_images = len(db)

    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(db, transformer, input_name, image_dims, crop_dims)

    # Now, build the region of interest and label blobs
    labels_blob = np.zeros((0), dtype=np.float32)
    for i in xrange(num_images):
        labels = db[i]['label']

        # Add to labels blob
        labels_blob = np.hstack((labels_blob, labels))

    # For debug visualizations
    # _vis_minibatch(im_blob, labels_blob)

    blobs = {
        'data': im_blob,
        'label': labels_blob
    }

    return blobs


def _get_image_blob(is_training, db, transformer, input_name, image_dims,
                    crop_dims):
    """Builds an input blob from the images in the db"""
    num_images = len(db)

    inputs = [
        caffe.io.load_image(db[i]['image'])
        for i in xrange(num_images)
    ]

    # Scale to standardize input dimensions.
    input_ = np.zeros(
        (len(inputs),
        image_dims[0],
        image_dims[1],
        inputs[0].shape[2]),
        dtype=np.float32
    )
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, image_dims)

    # TODO: We may include oversampling for validation?
    if is_training:
        # Take random crop.
        center = np.array([
            npr.uniform(
                crop_dims[i] / 2.0,
                image_dims[i] - crop_dims[i] / 2.0
            )
            for i in range(2)
        ])
    else:
        # Take center crop.
        center = np.array(image_dims) / 2.0

    crop_dims = np.array(crop_dims)
    crop = np.tile(center, (1, 2))[0] + np.concatenate([
        -crop_dims / 2.0,
        crop_dims / 2.0
    ])
    input_ = input_[:, crop[0]:crop[2], crop[1]:crop[3], :]

    if is_training:
        # random vertical flip
        if npr.choice([True, False]):
            input_ = input_[:, :, ::-1, :]

    # Preprocess and convert to Caffe format
    caffe_in = np.zeros(
        np.array(input_.shape)[[0, 3, 1, 2]],
        dtype=np.float32
    )
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = transformer.preprocess(input_name, in_)

    return caffe_in
