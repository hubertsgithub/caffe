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


def get_minibatch(db, is_training, num_classes, transformer, input_name,
                  image_dims, crop_dims, make_full_label_blob):
    """Given a db, construct a minibatch sampled from it."""
    num_images = len(db)

    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(
        db, 'image', is_training, transformer, input_name, image_dims,
        crop_dims
    )

    # Now, build the label blobs
    if make_full_label_blob:
        labels_blob = np.zeros((num_images, num_classes), dtype=np.float32)
        for i in xrange(num_images):
            label = db[i]['label']
            labels_blob[i, label] = 1
    else:
        labels_blob = np.zeros((num_images), dtype=np.float32)
        for i in xrange(num_images):
            labels_blob[i] = db[i]['label']

    # For debug visualizations
    # _vis_minibatch(im_blob, labels_blob)

    blobs = {
        input_name: im_blob,
        'label': labels_blob
    }

    return blobs


def get_tag_minibatch(db, is_training, tag_names, num_classes, transformer,
                      input_name, image_dims, crop_dims):
    """Given a db, construct a minibatch sampled from it."""
    num_images = len(db)

    # Get the input image blob, formatted for caffe
    im_blob = _get_image_blob(
        db, 'filepath', is_training, transformer, input_name, image_dims,
        crop_dims
    )

    blobs = {input_name: im_blob}

    # Now, build the tag blobs
    for tn, nc in zip(tag_names, num_classes):
        tags_blob = np.zeros((num_images, nc), dtype=np.float32)
        for i in xrange(num_images):
            tag_list = db[i]['tags_dic'][tn]
            tags_blob[i, tag_list] = 1

        blobs[tn] = tags_blob

    return blobs


def _get_image_blob(db, image_name, is_training, transformer, input_name,
                    image_dims, crop_dims):
    """Builds an input blob from the images in the db"""
    num_images = len(db)

    inputs = [
        caffe.io.load_image(db[i][image_name])
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
