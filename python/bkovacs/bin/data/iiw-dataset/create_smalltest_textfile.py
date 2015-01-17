import random

from lib.utils.misc.pathresolver import acrp
from lib.utils.data import fileproc

TESTFILEPATH = acrp('data/iiw-dataset/test.txt')
TESTSMALLFILEPATH = acrp('data/iiw-dataset/test-small.txt')
# number for each class
SAMPLECOUNT = 5000
CLASSNUMBER = 2


lines = fileproc.freadlines(TESTFILEPATH)
lines_by_class = [[] for c in range(CLASSNUMBER)]

for l in lines:
    tokens = l.split(' ')
    grayimg_path, chromimg_path, classid, p1x, p1y, p2x, p2y = tokens
    lines_by_class[int(classid)].append(l)

samples = []

for c in range(CLASSNUMBER):
    samples.append(random.sample(lines_by_class[c], SAMPLECOUNT))

zipped = zip(*samples)
sampled_lines = [e for t in zipped for e in t]

fileproc.fwritelines(TESTSMALLFILEPATH, sampled_lines)
