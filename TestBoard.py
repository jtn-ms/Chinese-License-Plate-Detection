#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  2 22:10:46 2017

@author: ubuntu

ref:http://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_contours/py_contour_features/py_contour_features.html?highlight=rectangle
    https://stackoverflow.com/questions/42235429/python-opencv-shape-detection
    https://stackoverflow.com/questions/41879315/opencv-using-cv2-approxpolydp-correctly
"""
from Segmentation import Segmentation
from ColorFilter import ColorFilter
import cv2
from skimage import io
import numpy as np
from Denoise import Denoise
from matplotlib import pyplot as plt
from RGB2HSV import checkBlue,checkYellow,rgb2hsv
##
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "1.jpg"
fullpath = dataset_path + filename
###

from RGB2HSV import opencv2skimage,skimage2opencv

def contour2box(contour):
    rect = cv2.minAreaRect(contour)
    box = cv2.boxPoints(rect)
    box = np.int0(box)
    return box

def estimateColr(img,
                 rect,
                 Debug=False):
    [x, y, w, h] = rect
    roi = img[y:y+h, x:x+w]
    rgb = cv2.cvtColor(roi,cv2.COLOR_BGR2RGB)
    totalcolr = [0,0,0]
    for j in range(h):
        for i in range(w):
            totalcolr += rgb[j,i]
    pixelcount = w * h
    avergecolr = totalcolr / pixelcount
    if Debug:
        print(avergecolr)
    hsv = rgb2hsv(avergecolr)
    return checkBlue(hsv) or checkYellow(hsv)
    
def ContourFiltering(img,
                     mask,
                     contours,
                     Debug=False):
    height,width = mask.shape
    refined = []
    for contour in contours:
        #perimeter = cv2.arcLength(contour,True)
        epsilon = 0.04*cv2.arcLength(contour,True)
        approx = cv2.approxPolyDP(contour,epsilon,True)
        boundingRect = cv2.boundingRect(contour)
        box = contour2box(contour)
        w = np.linalg.norm(box[0]-box[1])
        h = np.linalg.norm(box[1]-box[2])
        ratio = float(w) / h
        if Debug:
            print("ContourFiltering,approx,ratio:\n", approx,ratio)
        if len(approx) > 3 and len(approx) < 7 and\
            (ratio > 1.3 or ratio < 0.75) and \
            estimateColr(img,boundingRect):
            #areasize > (imgsize / 300) and\
            #areasize < (imgsize / 40) and \
            refined.append(contour)
            
    return refined

    '''
    boundingRect = cv2.boundingRect(contour)
    [intBoundingRectX, intBoundingRectY, intBoundingRectWidth, intBoundingRectHeight] = boundingRect
    intBoundingRectArea = intBoundingRectWidth * intBoundingRectHeight
    intCenterX = (intBoundingRectX + intBoundingRectX + intBoundingRectWidth) / 2
    intCenterY = (intBoundingRectY + intBoundingRectY + intBoundingRectHeight) / 2
    fltDiagonalSize = math.sqrt((intBoundingRectWidth ** 2) + (intBoundingRectHeight ** 2))
    fltAspectRatio = float(intBoundingRectWidth) / float(intBoundingRectHeight)
    '''
def findBBox(img,
             mask,
             Debug=False):
    # Dilate
    kernel = np.ones((3,3),np.uint8)
    ##opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)
    mask = cv2.dilate(mask,kernel,iterations=3)
    mask = cv2.erode(mask,kernel,iterations=1)
    if Debug:
        cv2.imshow("otsu",mask)
        key = cv2.waitKey(0)
        if key == 27:
            cv2.destroyAllWindows()
    # Find Contours
    imgContours,contours,hierarchy = cv2.findContours(mask, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE) 
    contours = ContourFiltering(img,mask,contours)
    if len(contours) == 0:
        return None
    
    cnt = contours[0]
    ##M = cv2.moments(cnt)
    # Rotated Rectangle
    box = contour2box(cnt)

    return box
        
def drawBBox(img,
            box,
            Debug = False):

    if box is None:
        print("BBox is None")
        return
    cv2.drawContours(img,[box],-1,(0,0,255),2)     
    # Direction Line ?x,y?y,x
    pt1 = (int((box[0][0] + box[3][0]) / 2),int((box[0][1] + box[3][1]) / 2))
    pt2 = (int((box[1][0] + box[2][0]) / 2),int((box[1][1] + box[2][1]) / 2))
    pt3 = (int((box[0][0] + box[1][0]) / 2),int((box[0][1] + box[1][1]) / 2))
    pt4 = (int((box[2][0] + box[3][0]) / 2),int((box[2][1] + box[3][1]) / 2))
    if np.linalg.norm(np.array(pt1)-np.array(pt2)) > np.linalg.norm(np.array(pt3)-np.array(pt4)):
        img = cv2.line(img,pt1,pt2,(0,255,0),2)
    else:
        img = cv2.line(img,pt3,pt4,(0,255,0),2)
    
    io.imshow(opencv2skimage(img))   
###        
def DetectLP1():
    # Load
    origin = cv2.imread(fullpath)
    # Denoise
    origin = Denoise(origin)
    # Resize
    h,w,c = origin.shape
    img = cv2.resize(origin,((w*200)/h,200))
    # Segmentation
    out1,out2 = Segmentation(img=opencv2skimage(img),Debug=True)
    # Blue Color Filter
    mask,res = ColorFilter(origin=skimage2opencv(out1),Debug=False)
    # Mark
    box = findBBox(skimage2opencv(img),mask,Debug=True)
    # Check Candidate
    drawBBox(img,box,Debug=True)

if __name__ == "__main__":
    DetectLP1()
