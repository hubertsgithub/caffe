#!/bin/bash

./build/tools/caffe visualize --model=examples/mnist/lenet_train_test.prototxt --weights=examples/mnist/lenet_iter_10000.caffemodel --datalayer=data --visualizedlayer=ip2 --datalayer_mean_to_add=0 --gradient_upscale=1000 --visualizationdir=examples/mnist/vis
