# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

"""Compute minibatch blobs for training a Fast R-CNN network."""

import json
import os
import time

import numpy as np
import numpy.random as npr

import caffe
import skimage.io


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
                      input_names, image_dims, crop_dims, make_one_tag_blob,
                      regression):
    """Given a db, construct a minibatch sampled from it."""
    num_images = len(db)

    blobs = {}
    for input_name in input_names:
        # For backward compatibility, we handle the one input case differently
        if len(input_names) == 1:
            image_key = 'filepath'
        else:
            image_key = 'filepath-{}'.format(input_name)

        # Note: There is only one transformer for all image inputs, so we just
        # use the first input_name to pass to the transformer
        im_blob = _get_image_blob(
            db, image_key, is_training, transformer, input_names[0],
            image_dims, crop_dims
        )
        blobs[input_name] = im_blob

    # Now, build the tag blobs
    for tn, nc in zip(tag_names, num_classes):
        if make_one_tag_blob:
            tags_blob = np.zeros((num_images, 1), dtype=np.float32)
            for i in xrange(num_images):
                # The label should be an integer value
                if tn not in db[i]['tags_dic']:
                    label = -1
                else:
                    label = int(db[i]['tags_dic'][tn])
                    if label < 0 or label >= nc:
                        raise ValueError(
                            'Label {} is out or range, number of classes: {}'.format(label, nc)
                        )
                tags_blob[i, 0] = label
        elif regression:
            tags_blob = np.zeros((num_images, 1), dtype=np.float32)
            for i in xrange(num_images):
                # The label should be an float value
                label = float(db[i]['tags_dic'][tn])
                tags_blob[i, 0] = label
        else:
            tags_blob = np.zeros((num_images, nc), dtype=np.float32)
            for i in xrange(num_images):
                tag_list = db[i]['tags_dic'][tn]
                # Set values to 1 where we have tags
                tags_blob[i, tag_list] = 1

        blobs[tn] = tags_blob

    #save_blobs_debug(input_names, transformer, blobs, make_one_tag_blob, regression)

    return blobs


def save_blobs_debug(input_names, transformer, blobs, make_one_tag_blob, regression):
    root_dir = 'python_data_layer_debug'
    if not os.path.exists(root_dir):
        os.mkdir(root_dir)

    print 'Saving debug images to {}'.format(os.path.abspath(root_dir))

    for blob_name, blob in blobs.iteritems():
        count = blob.shape[0]

        for i in range(count):
            filepath = os.path.join(root_dir, '{}-{}'.format(i, blob_name))
            if blob_name in input_names:
                blob_data = transformer.deprocess(input_names[0], blob[i])
            else:
                blob_data = blob[i]

            if blob_data.ndim >= 2:
                full_filepath = filepath + '.jpg'
                skimage.io.imsave(full_filepath, blob_data)
            else:
                # If these are tags, just list the indices
                #unique_items = np.unique(blob_data)
                #if np.array_equal(unique_items, [0, 1]) or np.array_equal(unique_items, [0]):
                if not make_one_tag_blob and not regression:
                    data_to_save = np.nonzero(blob_data)[0]
                else:
                    data_to_save = blob_data.tolist()
                json.dump(list(data_to_save), open(filepath + '.json', 'w'))


def _get_image_blob(db, image_key, is_training, transformer, input_name,
                    image_dims, crop_dims):
    """Builds an input blob from the images in the db"""
    verbose = False
    num_images = len(db)

    start = time.clock()
    inputs = [
        caffe.io.load_image(db[i][image_key])
        for i in xrange(num_images)
    ]
    elapsed = time.clock() - start
    if verbose:
        print 'Loading images took {:.2f} seconds.'.format(elapsed)

    # Scale to standardize input dimensions.
    start = time.clock()
    input_ = np.zeros(
        (
            len(inputs),
            image_dims[0],
            image_dims[1],
            inputs[0].shape[2]
        ),
        dtype=np.float32
    )
    for ix, in_ in enumerate(inputs):
        input_[ix] = caffe.io.resize_image(in_, image_dims)

    elapsed = time.clock() - start
    if verbose:
        print 'Resizing images took {:.2f} seconds.'.format(elapsed)

    # TODO: We take the same crop for all images...
    start = time.clock()
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
    elapsed = time.clock() - start
    if verbose:
        print 'Cropping images took {:.2f} seconds.'.format(elapsed)

    if is_training:
        # random vertical flip
        if npr.choice([True, False]):
            input_ = input_[:, :, ::-1, :]

    # Preprocess and convert to Caffe format
    caffe_in = np.zeros(
        np.array(input_.shape)[[0, 3, 1, 2]],
        dtype=np.float32
    )
    start = time.clock()
    for ix, in_ in enumerate(input_):
        caffe_in[ix] = transformer.preprocess(input_name, in_)
    elapsed = time.clock() - start
    if verbose:
        print 'Caffe preprocessing images took {:.2f} seconds.'.format(elapsed)

    return caffe_in
