#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 10:49:42 2017

@author: ubuntu
"""

from skimage import data, io, segmentation, color
from skimage.future import graph
import numpy as np
from matplotlib import pyplot as plt
from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.transform import resize
import cv2

from Denoise import Denoise

from RGB2HSV import rgb2hsv,opencv2skimage,checkBlue,EqualizeHist
#
dataset_path = "./sample/"
filename = "11.jpg"
fullpath = dataset_path + filename

def _weight_mean_color(graph, src, dst, n):
    """Callback to handle merging nodes by recomputing mean color.

    The method expects that the mean color of `dst` is already computed.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    n : int
        A neighbor of `src` or `dst` or both.

    Returns
    -------
    data : dict
        A dictionary with the `"weight"` attribute set as the absolute
        difference of the mean color between node `dst` and `n`.
    """
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
    """Callback called before merging two nodes of a mean color distance graph.

    This method computes the mean color of `dst`.

    Parameters
    ----------
    graph : RAG
        The graph under consideration.
    src, dst : int
        The vertices in `graph` to be merged.
    """
    graph.node[dst]['total color'] += graph.node[src]['total color']
    graph.node[dst]['pixel count'] += graph.node[src]['pixel count']
    graph.node[dst]['mean color'] = (graph.node[dst]['total color'] /
                                     graph.node[dst]['pixel count'])

def Segmentation(img = io.imread(fullpath),
                 Levels = 2,
                Level1 = "QuickShift",
                Level2 = "RAG_Merging",
                useBounday = False,
                Debug = False):
    # Level1
    if Level1 == "QuickShift":    
        labels1 = segmentation.quickshift(img, kernel_size=3, max_dist=10, ratio=0.7)#.slic(img, compactness=30, n_segments=400)
    elif Level1 == "SLIC":
        labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    elif Level1 == "felzenszwalb":
        labels1 = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    elif Level1 == "Watershed":
        gradient = sobel(rgb2gray(img))
        labels1 = segmentation.watershed(gradient, markers=250, compactness=0.001)
        
    out1 = color.label2rgb(labels1, img, kind='avg')

    if Levels == 1:
        if Debug:
            io.imshow(out1)
            print(labels1)
        return out1,labels1

    # Level2
    if Level2 == "NormalizedCut":        
        g = graph.rag_mean_color(img, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
    elif Level2 == "RAG_Thresholding":
        g = graph.rag_mean_color(img, labels1)
        labels2 = graph.cut_threshold(labels1, g, 29)
    elif Level2 == "RAG_Merging":
        g = graph.rag_mean_color(img, labels1)      
        labels2 = graph.merge_hierarchical(labels1, g, thresh=35, rag_copy=False,
                                           in_place_merge=True,
                                           merge_func=merge_mean_color,
                                           weight_func=_weight_mean_color)       
        
    out2 = color.label2rgb(labels2, img, kind='avg')
    
    if useBounday:
        out1 = segmentation.mark_boundaries(out1, labels1, (1, 0, 0))
        out2 = segmentation.mark_boundaries(out2, labels2, (1, 1, 0))

    if Debug:
        print(labels1)
        print(labels2)       
        fig, ax = plt.subplots(nrows=3, sharex=True, sharey=True, figsize=(10, 12))
        print('level1 segments: {}'.format(len(np.unique(labels1))))
        print('level2 segments: {}'.format(len(np.unique(labels2))))
        
        ax[0].imshow(img)
        ax[1].imshow(out1)
        ax[2].imshow(out2)
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
    
    return out1,out2

def Segmentation2(img = io.imread(fullpath),
                 Levels = 2,
                Level1 = "QuickShift",
                Level2 = "NormalizedCut",
                useBounday = False,
                Debug = False):
    # Level1
    if Level1 == "QuickShift":    
        labels1 = segmentation.quickshift(img, kernel_size=3, max_dist=10, ratio=0.7)#.slic(img, compactness=30, n_segments=400)
    elif Level1 == "SLIC":
        labels1 = segmentation.slic(img, compactness=30, n_segments=400)
    elif Level1 == "felzenszwalb":
        labels1 = segmentation.felzenszwalb(img, scale=100, sigma=0.5, min_size=50)
    elif Level1 == "Watershed":
        gradient = sobel(rgb2gray(img))
        labels1 = segmentation.watershed(gradient, markers=250, compactness=0.001)
        
    out1 = color.label2rgb(labels1, img, kind='avg')

    # Level2
    if Level2 == "NormalizedCut":        
        g = graph.rag_mean_color(img, labels1, mode='similarity')
        labels2 = graph.cut_normalized(labels1, g)
    elif Level2 == "RAG_Thresholding":
        g = graph.rag_mean_color(img, labels1)
        labels2 = graph.cut_threshold(labels1, g, 29)
        
    out2 = color.label2rgb(labels2, img, kind='avg')
    # Level3

    
    g2 = graph.rag_mean_color(img, labels2)      
    labels3 = graph.merge_hierarchical(labels2, g2, thresh=35, rag_copy=False,
                                       in_place_merge=True,
                                       merge_func=merge_mean_color,
                                       weight_func=_weight_mean_color)       
        
    out3 = color.label2rgb(labels3, img, kind='avg')
    
    if useBounday:
        out1 = segmentation.mark_boundaries(out1, labels1, (1, 0, 0))
        out2 = segmentation.mark_boundaries(out2, labels2, (0, 1, 0))
        out3 = segmentation.mark_boundaries(out2, labels3, (1, 1, 0))

    if Debug:      
        fig, ax = plt.subplots(nrows=4, sharex=True, sharey=True, figsize=(10, 12))
        print('level1 segments: {}'.format(len(np.unique(labels1))))
        print('level2 segments: {}'.format(len(np.unique(labels2))))
        print('level3 segments: {}'.format(len(np.unique(labels3))))
        
        ax[0].imshow(img)
        ax[1].imshow(out1)
        ax[2].imshow(out2)
        ax[3].imshow(out3)
        
        for a in ax:
            a.axis('off')
        
        plt.tight_layout()
    
    return out1,out2,out3

import time
if __name__ == "__main__":
    start = time.time() * 1000
    # Load
    origin = cv2.imread(fullpath)
    # Denoise
    origin = Denoise(origin)
    # Resize
    h,w,c = origin.shape
    img = cv2.resize(origin,(int(w*120/h),120))
    # equalization
    #img = EqualizeHist(img)
    # Blur
    blur = cv2.GaussianBlur(img,(7,7),3)
    # Segmentation
    out1,out2 = Segmentation(opencv2skimage(blur),
                                 Debug=True,
                                 Levels=2,
                                 useBounday = False)
    print("Total time spent : ",(time.time() * 1000 - start),"ms")