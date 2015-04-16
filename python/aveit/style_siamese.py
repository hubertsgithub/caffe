
# coding: utf-8

# ### Setup
# 
# The usual modules.

import numpy as np
import itertools as it
from tsne import bh_sne
from PIL import Image
import itertools
import scipy as sp
import math
import os
import os.path
import glob
import random
import sys

### Parameters to set
EXPDATAFOLDER = '../../experiments/clothingstyle/'
FILENAMES_FILE = 'categories_googlenet_sorted.npy'
FEATURES_FILE = 'features_googlenet_sorted.npy'
# look at dataanalysis.py for the categories and levels
CATEGORY = 'Shirts' #put 'all' for all categories
CATEGORYLEVEL = 3
IMG_SIZE = 160
SIZE_EMBEDDING = 100000
SCALAR = 200
RANDOM = True
IMAGE_FILENAME = 'embedding_{}_{}.jpg'.format(CATEGORY,SIZE_EMBEDDING)

def createImage_new(xs, ys, heights, widths, urls, binsize, img=None):
    min_x = np.min(xs)
    max_x = np.max(xs)
    min_y = np.min(ys)
    max_y = np.max(ys)
    width=max_x-min_x+binsize
    height=max_y-min_y+binsize
    img = np.ones((height, width, 3))
    for x, y, u  in it.izip(xs, ys, urls):
        im = load_image(u)
        im = resize_and_crop_image(im, binsize, binsize, keep_aspect_ratio=True, use_greater_side=True)
        # print im.shape
        x_off = 0
        y_off = 0
        if im.shape[0]<binsize:
            y_off = int((binsize-im.shape[0])/2.0)
            # img[y-min_y+int((binsize-im.shape[0])/2.0):y-min_y+int((binsize-im.shape[0])/2.0)+im.shape[0],x-min_x:x-min_x+binsize,:]=im
        if im.shape[1]<binsize:
            x_off = int((binsize-im.shape[1])/2.0)
            img[y-min_y:y-min_y+binsize,x-min_x+int((binsize-im.shape[1])/2.0):x-min_x+int((binsize-im.shape[1])/2.0)+im.shape[1],:]=im
        img[y-min_y+y_off:y-min_y+y_off+im.shape[0],x-min_x+x_off:x-min_x+x_off+im.shape[1],:]=im   
        # img[y-min_y:y-min_y+binsize,x-min_x:x-min_x+binsize,:]=im    

        
    return img

# ### Quantize Embedding
def quantize_embedding(embedding, binsize):
    quantized_space = {}
    for i, (x,y) in enumerate(embedding):
        qx,qy = binsize*int(x/binsize), binsize*int(y/binsize)
        quantized_space[(qx,qy)] = quantized_space.get((qx,qy),[]) + [i]
    return (quantized_space, binsize) 

def convert_to_quantized(data_files, (quantized_space, binsize)):
    urls = []
    xs = []
    ys = []
    heights = []
    widths = []
    for (x,y), imglist in quantized_space.items():
        which_img = np.random.choice(imglist)
        savename = '../../' + data_files[which_img]
        # im = load_image('../../' + data_files[which_img])
        # im = resize_and_crop_image(im, binsize, binsize, keep_aspect_ratio=False, use_greater_side=False)
        # savename='imgs/'+ str(x)+ str(y) +'.jpg'
        # save_image(savename, im)
        urls.append(savename)
        xs.append(x)
        ys.append(y)
        heights.append(binsize)
        widths.append(binsize)
    return xs, ys, urls, heights, widths

# ### Image Handling Helper Functions from Balazs
def resize_and_crop_image(arr, resize, crop, keep_aspect_ratio=False, use_greater_side=True):
    if arr.ndim == 3:
        rets = []
        for c in range(3):
            rets.append(resize_and_crop_channel(arr[:, :, c], resize, crop, keep_aspect_ratio, use_greater_side))
        res_arr = np.dstack(rets)
    elif arr.ndim == 2:
        res_arr = resize_and_crop_channel(arr, resize, crop, keep_aspect_ratio, use_greater_side)
    else:
        raise ValueError('The provided image array should have either 1 or 3 channels!')

    return res_arr
def load_image(filename):
    """ Load an image that is either linear or sRGB-encoded. """
    if not filename:
        raise ValueError('Empty filename')
    image = np.asarray(Image.open(filename)).astype(np.float) / 255.0
    return image
def save_image(filename, imagearr):
    """
    The image values should be in [0.0, 1.0]
    Save an image that is either linear or sRGB-encoded.
    """
    if not filename:
        raise ValueError('Empty filename')
    if not (imagearr.ndim == 2 or (imagearr.ndim == 3 and (imagearr.shape[2] == 1 or imagearr.shape[2] == 3))):
        raise ValueError('Invalid image dimensions: {0}'.format(imagearr.shape))
    if imagearr.ndim == 3 and imagearr.shape[2] == 1:
        imagearr = np.squeeze(imagearr, axis=2)
    imagearr *= 255.0
    imagearr = np.asarray(imagearr, dtype=np.uint8)
    image = Image.fromarray(imagearr)
    image.save(filename)

def compute_crop_tuple(width, height, cropw, croph, croplen):
    cropminus = math.floor(croplen / 2.)
    cropplus = math.ceil(croplen / 2.)
    fromw = int(max(0, cropw - cropminus))
    tow = int(min(width, cropw + cropplus))
    fromh = int(max(0, croph - cropminus))
    toh = int(min(height, croph + cropplus))
    tup = (fromw, fromh, tow, toh)
    return tup

def resize_and_crop_channel(ch_arr, resize, crop, keep_aspect_ratio=False, use_greater_side=True):
    '''
    Resizes and crops the middle of the provided image channel array
    '''
    if ch_arr.ndim != 2:
        raise ValueError('The provided image array should be two dimensional! Provided array dimensions: {0}'.format(ch_arr.shape))
    image = Image.fromarray(ch_arr)
    if resize is not None:
        if keep_aspect_ratio:
            w, h = image.size
            if (w > h and use_greater_side) or (w < h and not use_greater_side):
                r = float(resize) / w
                dim = (resize, int(h * r))
                image = image.resize(dim, Image.BILINEAR)
            else:
                r = float(resize) / h
                dim = (int(w * r), resize)
                image = image.resize(dim, Image.BILINEAR)
        else:
            image = image.resize((resize, resize), Image.BILINEAR)
    if crop is not None:
        w, h = image.size
        middle = [x / 2 for x in image.size]
        tup = compute_crop_tuple(w, h, middle[0], middle[1], crop)
        image = image.crop(tup)
    ret = np.array(image)
    return ret

if __name__ == '__main__':
    #load file names
    data_files = np.load(EXPDATAFOLDER + FILENAMES_FILE)

    alength = data_files.shape[0]
    features = np.load(EXPDATAFOLDER + FEATURES_FILE)
    blength = features.shape[0]
    max_len = min(alength,blength)
    data_files = data_files[0:max_len,:]
    features = features[:max_len]

    #Select individual classes
    if CATEGORY != 'all':
        indices = np.where(data_files[:,CATEGORYLEVEL]==CATEGORY)
        data_files = data_files[indices]
        features = features[indices]

    #generate random subset of files
    max_len = min(max_len, features.shape[0])
    size = min(features.shape[0], SIZE_EMBEDDING)
    data_files = data_files[:,0]

    if RANDOM == True:
        nums = np.zeros(max_len)
        nums[:size] = 1
        np.random.shuffle(nums)
        np.save('../../experiments/clothingstyle/rand_subset_{}_{}.npy'.format(SIZE_EMBEDDING,CATEGORY), nums)
        # nums = np.load('../../experiments/clothingstyle/rand_subset_{0}.npy'.format(SIZE_EMBEDDING))
        data_files = data_files[nums==1]
        #load features
        features = features[nums==1,:].astype('float64')
        #compute embedding
        X_2 = bh_sne(features)
        np.save('../../experiments/clothingstyle/image_tsne_embedding_Google_Siamese_rand_{}_{}.npy'.format(SIZE_EMBEDDING,CATEGORY), X_2)
    else:
        nums = np.load('../../experiments/clothingstyle/rand_subset_{0}.npy'.format(SIZE_EMBEDDING))
        data_files = data_files[nums==1]
        #load features
        features = features[nums==1,:].astype('float64') 
        X_2 = np.load('../../experiments/clothingstyle/image_tsne_embedding_Google_Siamese_rand_{}_{}.npy'.format(SIZE_EMBEDDING,CATEGORY))

        # X_2 = np.load('../../experiments/clothingstyle/image_tsne_embedding_Google_Siamese_rand_100000.npy')

    #scale embedding
    X_2b = X_2*SCALAR

    xs,ys,urls, heights, widths = convert_to_quantized(data_files, quantize_embedding(X_2b, IMG_SIZE))

    img = createImage_new(xs, ys, heights, widths, urls, IMG_SIZE)

    save_image(EXPDATAFOLDER + IMAGE_FILENAME, img)


