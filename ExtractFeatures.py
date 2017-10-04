#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 06:40:47 2017

@author: ubuntu
"""

import numpy as np
import cv2
from matplotlib import pyplot as plt
from Denoise import Denoise
import numpy
##
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "11.jpg"
fullpath = dataset_path + filename
###
needNormalize = True
###
def extractROI(img):
    height, width, numChannels = img.shape
    x1 = int(width * 0.2)
    x2 = int(width * 0.8)
    y1 = int(height * 0.25)
    y2 = int(height * 0.75)
    return img[y1:y2,x1:x2,:]

def FeatureFiltering(pts,
                     ratio = 1.0,
                     filterRange = 20):

    refined = []#corners

    for pt in pts:
        count = 0
        for pt_ in pts:
            if numpy.linalg.norm(pt[0]-pt_[0]) < 30:
                count += 1
        if count > 5:
            refined.append(pt*ratio)
    return refined
    
def goodFeatures(gray,
                 numberoffeatures = 100,
                 qualityLevel = 0.01,
                 minDistance = 10):
    
    corners = cv2.goodFeaturesToTrack(gray,numberoffeatures,qualityLevel,minDistance)
    corners = np.int0(corners)
    
    return corners

def FeatureSpace(img):
    # RGB->HSV
    hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(hsv)
    blur = cv2.GaussianBlur(s,(5,5),0)
    #gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #s_ = np.array(s.shape)
    #s_ = cv2.normalize(s,  s_, 0, 255, cv2.NORM_MINMAX)
    return blur

def refinedGoodFeatures(colr,
                        tgt,
                        Debug=False):
    size=500.0
    h,w,c = colr.shape
    img = cv2.resize(colr,(int(w*size/h),int(size)))
    img = Denoise(img)
    fspace = FeatureSpace(img)         # preprocess to get grayscale and threshold images
    corners =  goodFeatures(fspace)
    refined = FeatureFiltering(corners,ratio=tgt.shape[0]/size)
    if Debug:
        checkFeatures(tgt,refined)
        
    return refined

def SIFT(gray):

    sift = cv2.SIFT()
    kp = sift.detect(gray,None)
    return kp

def checkFeatures(img,
                  corners,
                  Debug=False):
    
    mark = img.copy()
    mark[mark>=0] = 0
    #img = cv2.drawKeypoints(img,corners)
    
    for i in corners:
        x,y = i.ravel()
        cv2.circle(mark,(int(x),int(y)),2,(255,255,255),5)

    if Debug:
        cv2.imshow("goodFeatures",mark)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
    
    return mark
    
def main():
    # Load
    origin = cv2.imread(fullpath)
    # Resize
    h,w,c = origin.shape
    img = cv2.resize(origin,((w*500)/h,500))
    # Extract ROI
    #img = extractROI(img)
    '''
    # Denoise
    img = Denoise(img)
    # Grayscale & Normalize
    if needNormalize == True:
        gray = FeatureSpace(img)         # preprocess to get grayscale and threshold images
    else:
        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    # Extract Features
    corners = goodFeatures(gray)
    # Refine Features
    refined =  FeatureFiltering(corners)
    # Show Result
    checkFeatures(img,refined)
    '''
    refinedGoodFeatures(img,np.zeros((200,(w*200)/h,3)),Debug=True)
   
if __name__ == "__main__":
    main()