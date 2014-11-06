#!/bin/bash

./build/tools/caffe train -solver ownmodels/mitintrinsic/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel #-gpu 0
