import os

def get_caffe_root_path():
    curdir = __file__
    upcount = 6

    for i in range(upcount):
        curdir = os.path.dirname(curdir)

    return curdir

# add caffe root path
def acrp(p):
    return os.path.join(get_caffe_root_path(), p)
