"""An evaluation layer which can compute the label ranking average precision
It is useful in tag prediction experiments.
"""

import json

import numpy as np
from sklearn import metrics

import caffe


class RankAveragePrecisionLayer(caffe.Layer):
    """Layer which computes the label ranking average precision."""

    def __init__(self, layer_param):
        super(RankAveragePrecisionLayer, self).__init__(layer_param)

    def setup(self, bottom, top):
        top[0].reshape(1)

    def forward(self, bottom, top):
        """Compute the label ranking average precision."""
        y_score = bottom[0].data
        y_true = bottom[1].data
        label_rank_avg_prec = metrics.label_ranking_average_precision_score(y_true, y_score)

        top[0].data[...] = label_rank_avg_prec

    def backward(self, top, propagate_down, bottom):
        """This layer does not propagate gradients."""
        pass

    def reshape(self, bottom, top):
        # Single value output
        top[0].reshape(1)
        pass
