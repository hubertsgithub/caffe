#!/bin/bash
# This script converts the MIT intrinsic images dataset images to $RES x $RES resolution

DATA=data
FILTEREDDATA=filtereddata
DATAFOLDER=data/mitintrinsic

echo "Removing old filtered images..."
rm -rf $FILTEREDDATA

echo "Filtering images..."
mkdir $FILTEREDDATA

for dir in $DATA/*; do
    echo "Filtering images in $dir..."
    dirname=$(basename "$dir")
    echo "$DATAFOLDER/$FILTEREDDATA/$dirname-original.png $DATAFOLDER/$FILTEREDDATA/$dirname-original.png $DATAFOLDER/$FILTEREDDATA/$dirname-mask.png" >> $FILTEREDDATA/train.txt
    cp $dir/original.png $FILTEREDDATA/$dirname-original.png
    cp $dir/shading.png $FILTEREDDATA/$dirname-shading.png
    cp $dir/mask.png $FILTEREDDATA/$dirname-mask.png
done

echo "Done."
