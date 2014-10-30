#!/bin/bash
# Create the imagenet lmdb inputs
# N.B. set the path to the imagenet train + val data dirs

CAFFEHOME=/media/balazs/Data/Cornell/caffe
NAME=mitintrinsic
MODEL=$CAFFEHOME/ownmodels/$NAME
DATA=$CAFFEHOME/data/$NAME/filtereddata
TOOLS=$CAFFEHOME/build/tools

TRAIN_DATA_ROOT=$DATA/train/
VAL_DATA_ROOT=$DATA/val/

# Set RESIZE=true to resize the images to 256x256. Leave as false if images have
# already been resized using another tool.
RESIZE=true
if $RESIZE; then
  RESIZE_HEIGHT=256
  RESIZE_WIDTH=256
else
  RESIZE_HEIGHT=0
  RESIZE_WIDTH=0
fi

if [ ! -d "$TRAIN_DATA_ROOT" ]; then
  echo "Error: TRAIN_DATA_ROOT is not a path to a directory: $TRAIN_DATA_ROOT"
  echo "Set the TRAIN_DATA_ROOT variable in create_$NAME\.sh to the path" \
       "where the $NAME training data is stored."
  exit 1
fi

if [ ! -d "$VAL_DATA_ROOT" ]; then
  echo "Error: VAL_DATA_ROOT is not a path to a directory: $VAL_DATA_ROOT"
  echo "Set the VAL_DATA_ROOT variable in create_$NAME\.sh to the path" \
       "where the $NAME validation data is stored."
  exit 1
fi

echo "Creating train lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_$NAME\_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $TRAIN_DATA_ROOT \
    $DATA/train.txt \
    $MODEL/$NAME\_train_lmdb

echo "Creating val lmdb..."

GLOG_logtostderr=1 $TOOLS/convert_$NAME\_imageset \
    --resize_height=$RESIZE_HEIGHT \
    --resize_width=$RESIZE_WIDTH \
    --shuffle \
    $VAL_DATA_ROOT \
    $DATA/val.txt \
    $MODEL/$NAME\_val_lmdb

echo "Done."
