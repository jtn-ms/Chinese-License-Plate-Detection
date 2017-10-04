#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:39:32 2017

@author: ubuntu
"""

##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "2.jpg"
fullpath = dataset_path + filename
 
import matplotlib.pyplot as plt
import numpy as np

from skimage.data import astronaut
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import felzenszwalb, slic, quickshift, watershed
from skimage.segmentation import mark_boundaries
from skimage.util import img_as_float
from skimage import io
from skimage import data, segmentation, filters, color
from skimage.future import graph

import cv2

from Denoise import Denoise
from RGB2HSV import rgb2hsv,opencv2skimage,skimage2opencv,checkBlue,EqualizeHist
from ExtractFeatures import refinedGoodFeatures,checkFeatures
from ColorFilter import ColorFilter
from TestBoard import findBBox,drawBBox
import time

def _weight_mean_color(graph, src, dst, n):
    # color filtering
    hsv = rgb2hsv(graph.node[dst]['mean color'])
    hsv_ = rgb2hsv(graph.node[n]['mean color'])
    bluecnt = 0
    if checkBlue(hsv):
        bluecnt += 1
    if checkBlue(hsv_):
        bluecnt += 1
    if bluecnt == 2:
        diff = graph.node[dst]['mean color'] - graph.node[n]['mean color']
        diff = np.linalg.norm(diff)          
    elif bluecnt == 1:
        diff = 1000
    else:
        diff = 0
    
    return {'weight': diff}
    
def merge_mean_color(graph, src, dst):
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

def showResult(img,
               labels):
    out = color.label2rgb(labels, img, kind='avg')
    print('level1 segments: {}'.format(len(np.unique(labels))))
    #mark = segmentation.mark_boundaries(out, labels, (1, 0, 0))
    fig, ax = plt.subplots(ncols=2, sharex=True, sharey=True, figsize=(10, 12))
    
    ax[0].imshow(img)
    ax[1].imshow(out)
    #ax[2].imshow(mark)
    
    for a in ax:
        a.axis('off')
    
    plt.tight_layout()
    
    return out

def seg(img,
        Debug=False):
    # Level 1
    start = time.time() * 1000
    Level = "QuickShift"
    if Level == "QuickShift":    
        labels = segmentation.quickshift(img, kernel_size=3, max_dist=10, ratio=1.0)#.slic(img, compactness=30, n_segments=400)
    elif Level == "SLIC":
        labels = segmentation.slic(img, compactness=30, n_segments=400)
    elif Level == "felzenszwalb":
        labels = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    elif Level == "Watershed":
        gradient = sobel(rgb2gray(img))
        labels = segmentation.watershed(gradient, markers=250, compactness=0.001)
    print "level1 took ",int(time.time() * 1000 - start),"ms"
    # Show Result
    if Debug:
        showResult(img,labels)
    return labels
           
def refineLabels(labels,
                 corners,
                 howmany=20):
    # pick up labels on corners
    containcorners = []
    for corner in corners:
        x = int(corner[0][0])
        y = int(corner[0][1])
        containcorners.append(labels[y][x])
    # make table(corners,label)
    match = []
    for label in np.unique(containcorners):
        match.append([(containcorners == label).sum(),label])
        
    match = sorted(match,reverse=True)
    match = np.array(match)
    print(match.shape)
    howmany = howmany if howmany < match.shape[0] else match.shape[0]
    # select labels with enough corners    
    pickup = []
    for i in range(howmany):
        pickup.append(match[i][1])
    # mark bg with 0
    mark = 0
    labels_ = np.zeros(labels.shape)
    #
    for label in np.unique(pickup):
        mark += 1
        labels_[labels == label] = mark
    
    return labels_

GAUSSIAN_SMOOTH_FILTER_SIZE = (3, 3)
ADAPTIVE_THRESH_BLOCK_SIZE = 7
ADAPTIVE_THRESH_WEIGHT = 3

def changebgcolr(img,
                 labels,
                 Debug=False):
    h,w,c = img.shape
    for j in range(h):
        for i in range(w):
            if labels[j][i] == 0:
                for k in range(3):
                    img[j][i][k] = 0
    if Debug:
        cv2.imshow("img",skimage2opencv(img))
        cv2.waitKey(10000)
        cv2.destroyAllWindows()
        
def DetectLP(path):
    # Load
    origin = cv2.imread(path)
    # Resize
    h,w,c = origin.shape
    img = cv2.resize(origin,((w*200)/h,200))
    # Blur
    blur = cv2.GaussianBlur(img,(5,5),3)
    
    # Extract Good Features
    corners = refinedGoodFeatures(origin,img)
    featureimg = checkFeatures(img,corners)
    # Denoise
    origin = Denoise(origin)
    # Opencv2Skimage
    skimg = cv2.cvtColor(blur, cv2.COLOR_BGR2RGB)
    # Segmentation
    labels = seg(skimg)
    # Eval Label
    labels = refineLabels(labels,corners,howmany=20)
    # Show Result
    out = showResult(skimg,labels)
    changebgcolr(out,labels)
    # Blue Color Filter
    gray = cv2.cvtColor(skimage2opencv(out),cv2.COLOR_BGR2GRAY)
    ret, mask = cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #ret1, mask = cv2.threshold(mask,0,255,cv2.THRESH_BINARY)
    gray2 = cv2.cvtColor(featureimg,cv2.COLOR_BGR2GRAY)
    ret2,fmask = cv2.threshold(gray2,0,255,cv2.THRESH_BINARY)
    #mask = cv2.min(fmask,mask)
    # Find Candidate
    box = findBBox(img,mask)   
    # Check Candidate
    drawBBox(img,box,Debug=True)

if __name__ == "__main__":
    DetectLP(fullpath)