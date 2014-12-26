#!/bin/bash

sudo apt-get -y remove libopenblas-base
sudo pip install pypng pyamg redis msgpack-numpy
make pycaffe
cd python/bkovacs
celery -A lib.intrinsic.tasks worker --loglevel=info

