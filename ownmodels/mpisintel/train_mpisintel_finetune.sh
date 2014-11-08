#!/bin/bash

./build/tools/caffe train -solver ownmodels/mpisintel/solver.prototxt -weights models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel #-gpu 0
