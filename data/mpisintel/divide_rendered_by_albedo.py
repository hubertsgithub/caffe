from os import listdir
from os.path import exists
import os
import cv2
import shutil

origpath = 'data/mpisintel/data/clean_noshadingtextures'
albedopath = 'data/mpisintel/data/albedo_noshadingtextures'
shadingpath = 'data/mpisintel/data/gen_shading'

# remove shadingpath, we will regenerate it
if exists(shadingpath):
    shutil.rmtree(shadingpath)

os.mkdir(shadingpath)

origdirnames = listdir(origpath)
origdirnames.sort()

first = True
for dir in origdirnames:
    origparentdirpath = origpath + '/' + dir
    albedoparentdirpath = albedopath + '/' + dir
    shadingparentdirpath = shadingpath + '/' + dir

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    if not exists(albedoparentdirpath):
        print albedoparentdirpath + " doesn't exist, moving on..."
        continue

    os.mkdir(shadingparentdirpath)

    origfilenames = listdir(origparentdirpath)
    origfilenames.sort()
    albedofilenames = listdir(albedoparentdirpath)
    albedofilenames.sort()

    print 'Processing {0}...'.format(origparentdirpath)
    for i in range(len(origfilenames)):
        origfilepath = origparentdirpath + '/' + origfilenames[i]
        albedofilepath = albedoparentdirpath + '/' + albedofilenames[i]

        if not exists(origfilepath):
            print origfilepath + " doesn't exist, moving on..."
            continue

        if not exists(albedofilepath):
            print albedofilepath + " doesn't exist, moving on..."
            continue

        origimage = cv2.imread(origfilepath)
        origimage = origimage.astype(float)
        origimage = origimage / 255.0

        albedoimage = cv2.imread(albedofilepath)
        albedoimage = albedoimage.astype(float)
        albedoimage = albedoimage / 255.0
        albedoimage = albedoimage + 0.001

        shadingimage = origimage / albedoimage

        shadingimage = shadingimage * 255.0

        cv2.imwrite(shadingparentdirpath + '/' + origfilenames[i], shadingimage)

print "Done!"
