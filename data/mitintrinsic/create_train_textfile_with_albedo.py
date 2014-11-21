from os import listdir
from os.path import exists
import os
import cv2

origpath = 'data/mitintrinsic/data'
resize = 190
crop = 190

origdirnames = listdir(origpath)
origdirnames.sort()

f = open('data/mitintrinsic/val_with_albedo.txt', 'w')

first = True
for dir in origdirnames:
    origparentdirpath = origpath + '/' + dir

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    filepaths = []
    filepaths.append(origparentdirpath + '/' + 'diffuse.png')
    filepaths.append(origparentdirpath  + '/' + 'shading.png')
    filepaths.append(origparentdirpath + '/' + 'reflectance.png')
    filepaths.append(origparentdirpath + '/' + 'mask.png')

    convertedfilepaths = []
    for filepath in filepaths:
        if not exists(filepath):
            print filepath + " doesn't exist, moving on..."
            continue

        image = cv2.imread(filepath)
        r = float(resize) / image.shape[0]
        dim = (int(image.shape[1] * r), resize)
        # perform the actual resizing of the image and show it
        resized = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
        middle = [x / 2 for x in resized.shape]
        fromw = max(0, middle[0] - crop / 2)
        tow = min(resized.shape[0], middle[0] + crop / 2)
        fromh = max(0, middle[1] - crop / 2)
        toh = min(resized.shape[1], middle[1] + crop / 2)

        fileName, fileExtension = os.path.splitext(filepath)
        convertedfilepath = fileName + '-converted' + fileExtension
        convertedfilepaths.append(convertedfilepath)
        cv2.imwrite(convertedfilepath, resized[fromw:tow, fromh:toh])

    f.write('{0} {1} {2} {3} {4}\n'.format(convertedfilepaths[0], convertedfilepaths[0], convertedfilepaths[1], convertedfilepaths[2], convertedfilepaths[3]))

    if first:
        f = open('data/mitintrinsic/train_with_albedo.txt', 'w')
        first = False

f.close()

print "Done!"
