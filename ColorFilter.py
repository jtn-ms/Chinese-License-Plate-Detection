#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 23:31:09 2017

@author: ubuntu

ref:https://www.pyimagesearch.com/2014/08/04/opencv-python-color-detection/
    http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=rectangle
"""

import os
import numpy as np
import cv2

##
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "35.jpg"
fullpath = dataset_path + filename
###

# define range of blue color in HSV
lower = np.array([110,70,50])
upper = np.array([130,255,255])
'''
# define range of red color in HSV
lower = np.array([165,30,20])#([30,150,50])
upper = np.array([180,120,120])#([255,255,180])
'''
# Set up the detector with default parameters.
# Setup SimpleBlobDetector parameters.
params = cv2.SimpleBlobDetector_Params()
# Change thresholds
params.minThreshold = 10;
params.maxThreshold = 200;
# Filter by Area.
params.filterByArea = True
params.minArea = 450
# Filter by Circularity
params.filterByCircularity = True
params.minCircularity = 0.5
# Filter by Convexity
params.filterByConvexity = True
params.minConvexity = 0.5
# Filter by Inertia
params.filterByInertia = True
params.minInertiaRatio = 0.05
'''
params.filterBySize = True#
params.minSize = 10
'''
def ColorFilter(origin = cv2.imread(fullpath),
                Resize = False,
                Debug = False):
    
    h,w,c = origin.shape
    if Resize:
        frame = cv2.resize(origin,((w*300)/h,300))
    else:
        frame = origin
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    res = cv2.bitwise_and(frame,frame, mask= mask)

    if Debug:
        cv2.imshow('frame',frame)
        cv2.imshow('mask',mask)
        cv2.imshow('res',res)
        
        k = cv2.waitKey(0)
        if k == 27:         # wait for ESC key to exit
            cv2.destroyAllWindows()
    
    return mask,res
    
def main():  
    path, fname = os.path.split(os.path.realpath("__file__"))
    image_dir = path + '\images'
    out_dir = path + '\out'
    
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
        
    for img_fn in os.listdir(image_dir):
        
        frame = cv2.imread(image_dir + '\\' + img_fn)
        
        #cv2.normalize(frame, frame, 0, 255, cv2.NORM_MINMAX)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        #color mask
        mask = cv2.inRange(hsv, lower, upper)
        res = cv2.bitwise_and(frame,frame, mask= mask)
        
        #smooth
        kernel = np.ones((30,30),np.float32)/225
        smoothed = cv2.filter2D(res,-1,kernel)
        
        #median
        median = cv2.medianBlur(res,15)
        
        #blob
        im = cv2.cvtColor(smoothed,cv2.COLOR_BGR2GRAY)
        ret,im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        ret,im = cv2.threshold(im,127,255,cv2.THRESH_BINARY_INV)
        # Create a detector with the parameters
        ver = (cv2.__version__).split('.')
        if int(ver[0]) < 3 :
            detector = cv2.SimpleBlobDetector(params)
        else : 
            detector = cv2.SimpleBlobDetector_create(params)
        # Detect blobs.
        keypoints = detector.detect(im)
         
        # Draw detected blobs as red circles.
        # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
        im_with_keypoints = cv2.drawKeypoints(frame, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        
        '''
        cv2.imshow('frame',cv2.resize(frame, (0,0), fx=0.2, fy=0.2))
        cv2.imshow('mask',cv2.resize(mask, (0,0), fx=0.2, fy=0.2))
        cv2.imshow('res',cv2.resize(res, (0,0), fx=0.2, fy=0.2))
        cv2.imshow('smooth',cv2.resize(smoothed, (0,0), fx=0.2, fy=0.2))
        cv2.imshow('median',cv2.resize(median, (0,0), fx=0.2, fy=0.2))
        cv2.imshow('keypoint',cv2.resize(im_with_keypoints, (0,0), fx=0.2, fy=0.2))
        #cv2.waitKey(0)
        '''
        cv2.imwrite(out_dir + "\\" + img_fn,im_with_keypoints)
        
if __name__ == "__main__":
    ColorFilter()