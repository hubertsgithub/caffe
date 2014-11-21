#!/bin/bash
# This script converts the MIT intrinsic images dataset images to $RES x $RES resolution

DATA=data
FILTEREDDATA=filtereddata
DATAFOLDER=data/mitintrinsic

echo "Removing old filtered images..."
rm -rf $FILTEREDDATA

echo "Filtering images..."
mkdir $FILTEREDDATA

first=true
for dir in $DATA/*; do
    echo "Filtering images in $dir..."
    dirname=$(basename "$dir")
	# Let the first directory be the validation set
	if [ $first = "true" ]; then
		echo "$DATAFOLDER/$FILTEREDDATA/$dirname-diffuse.png $DATAFOLDER/$FILTEREDDATA/$dirname-shading.png $DATAFOLDER/$FILTEREDDATA/$dirname-mask.png" >> $FILTEREDDATA/val.txt
		first=false
	else
		echo "$DATAFOLDER/$FILTEREDDATA/$dirname-diffuse.png $DATAFOLDER/$FILTEREDDATA/$dirname-shading.png $DATAFOLDER/$FILTEREDDATA/$dirname-mask.png" >> $FILTEREDDATA/train.txt
	fi

    cp $dir/diffuse.png $FILTEREDDATA/$dirname-diffuse.png
    cp $dir/shading.png $FILTEREDDATA/$dirname-shading.png
    cp $dir/mask.png $FILTEREDDATA/$dirname-mask.png
done

echo "Done."
