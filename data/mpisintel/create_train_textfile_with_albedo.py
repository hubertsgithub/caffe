from os import listdir
from os.path import exists

origpath = 'data/mpisintel/data/clean_noshadingtextures'
shadingpath = 'data/mpisintel/data/gen_shading'
albedopath = 'data/mpisintel/data/albedo_noshadingtextures'

origdirnames = listdir(origpath)
origdirnames.sort()

f = open('data/mpisintel/val_with_albedo.txt', 'w')

first = True
for dir in origdirnames:
    origparentdirpath = origpath + '/' + dir
    shadingparentdirpath = shadingpath + '/' + dir
    albedoparentdirpath = albedopath + '/' + dir

    if not exists(origparentdirpath):
        print origparentdirpath + " doesn't exist, moving on..."
        continue

    if not exists(shadingparentdirpath):
        print shadingparentdirpath + " doesn't exist, moving on..."
        continue

    if not exists(albedoparentdirpath):
        print albedoparentdirpath + " doesn't exist, moving on..."
        continue

    origfilenames = listdir(origparentdirpath)
    origfilenames.sort()
    shadingfilenames = listdir(shadingparentdirpath)
    shadingfilenames.sort()
    albedofilenames = listdir(albedoparentdirpath)
    albedofilenames.sort()

    for i in range(len(origfilenames)):
        origfilepath = origparentdirpath + '/' + origfilenames[i]
        shadingfilepath = shadingparentdirpath + '/' + shadingfilenames[i]
        albedofilepath = albedoparentdirpath + '/' + albedofilenames[i]

        if not exists(origfilepath):
            print origfilepath + " doesn't exist, moving on..."
            continue

        if not exists(shadingfilepath):
            print shadingfilepath + " doesn't exist, moving on..."
            continue

        if not exists(albedofilepath):
            print albedofilepath + " doesn't exist, moving on..."
            continue

        f.write('{0} {1} {2} {3}\n'.format(origfilepath, origfilepath, shadingfilepath, albedofilepath))

    if first:
        f = open('data/mpisintel/train_with_albedo.txt', 'w')
        first = False

f.close()

print "Done!"
