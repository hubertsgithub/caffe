
# coding: utf-8

# ### Setup
# 
# The usual modules.

import numpy as np
from PIL import Image
import itertools as it
import scipy as sp
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
# import sklearn.metrics.pairwise
from sklearn.metrics.pairwise import euclidean_distances
import math
import os
import os.path
import random
import sys

### Parameters to set
EXPDATAFOLDER = '../../experiments/clothingstyle/'
FILENAMES_FILE = 'categories_googlenet_sorted.npy'
FEATURES_FILE = 'features_googlenet_sorted.npy'
CATEGORY_A = 'Shoes'
CATEGORYLEVEL_A = 2 
CATEGORY_B = 'Dresses'
CATEGORYLEVEL_B = 3
IMG_SIZE = 160
NUMBEROFLINES = 10
NUMBEROFCLUSTERS = 20
STEPS = 10
IMAGE_FILENAME = 'abstracts_{}-{}_numlines_{}.jpg'.format(CATEGORY_A,CATEGORY_B,NUMBEROFLINES)



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

def convert_to_image_coord(data_files, lines, binsize):
    urls = []
    xs = []
    ys = []
    heights = []
    widths = []
    row = 0 
    for line in lines:
        col = 0
        for image in line:
            urls.append('../../' + image)
            xs.append(col * binsize)
            ys.append(row * binsize)
            heights.append(binsize)
            widths.append(binsize)
            col += 1
        row += 1

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

    #Select individual classes A
    indices = np.where(data_files[:,CATEGORYLEVEL_A]==CATEGORY_A)
    data_files_A = data_files[indices]
    features_A = features[indices]

    #Select individual classes B
    indices = np.where(data_files[:,CATEGORYLEVEL_B]==CATEGORY_B)
    data_files_B = data_files[indices]
    features_B = features[indices]

    k_means_A = KMeans(n_clusters=NUMBEROFCLUSTERS, init='k-means++', n_init=10)
    k_means_A.fit(features_A)
    centroids_A = k_means_A.cluster_centers_

    k_means_B = KMeans(n_clusters=NUMBEROFCLUSTERS, init='k-means++', n_init=10)
    k_means_B.fit(features_B)
    centroids_B = k_means_B.cluster_centers_

    distance_matrix = euclidean_distances(centroids_A, centroids_B)

    #remove categories from file list
    data_files_A = data_files_A[:,0]
    data_files_B = data_files_B[:,0]
    tree_A = KDTree(features_A, leaf_size=2)
    tree_B = KDTree(features_B, leaf_size=2)

    d_m_sorted = np.ravel(distance_matrix).argsort()
    
    close_pairs = np.unravel_index(d_m_sorted[:5], distance_matrix.shape)
    distant_pairs = np.unravel_index(d_m_sorted[:5:-1], distance_matrix.shape)

    num_examples = 3
    lines = []
    for i in range(NUMBEROFLINES):
        line = []
        point_A = centroids_A[close_pairs[0][i]]
        point_B = centroids_B[close_pairs[1][i]]
        dist, ind = tree_A.query(point_A, k=num_examples)
        line.extend(list(data_files_A[ind[0][0:num_examples]]))
        dist, ind = tree_B.query(point_B, k=num_examples)
        line.extend(list(data_files_B[ind[0][0:num_examples]]))
        lines.append(line)
    for i in range(NUMBEROFLINES):
        line = []
        point_A = centroids_A[distant_pairs[0][i]]
        point_B = centroids_B[distant_pairs[1][i]]
        dist, ind = tree_A.query(point_A, k=num_examples)
        line.extend(list(data_files_A[ind[0][0:num_examples]]))
        dist, ind = tree_B.query(point_B, k=num_examples)
        line.extend(list(data_files_B[ind[0][0:num_examples]]))
        lines.append(line)

    xs,ys,urls, heights, widths = convert_to_image_coord(data_files, lines, IMG_SIZE)

    img = createImage_new(xs, ys, heights, widths, urls, IMG_SIZE)

    save_image(EXPDATAFOLDER + IMAGE_FILENAME, img)



