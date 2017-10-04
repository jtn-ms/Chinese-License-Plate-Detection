#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Oct  3 10:45:51 2017

@author: ubuntu
"""

import cv2
import numpy as np
##### Test Variable
dataset_path = "/media/ubuntu/Investigation/DataSet/Image/Classification/Insurance/Insurance/Tmp/LP/"
filename = "1.jpg"
fullpath = dataset_path + filename

def main():
    img = cv2.imread(fullpath)
    height, width, numChannels = img.shape
    hsv = np.zeros((height, width, 3), np.uint8)
    luv = np.zeros((height, width, 3), np.uint8)
    HSL = np.zeros((height, width, 3), np.uint8)
    LAB = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    HSL = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    luv = cv2.cvtColor(img,cv2.COLOR_BGR2LUV)
    LAB = cv2.cvtColor(img,cv2.COLOR_BGR2LAB)
    YCrCb = cv2.cvtColor(img,cv2.COLOR_BGR2YCR_CB)
    h, s, v = cv2.split(hsv)
    l,u,v_ = cv2.split(luv)
    L,A,B = cv2.split(LAB)
    H,S,L = cv2.split(HSL)
    Y,Cr,Cb = cv2.split(YCrCb)
    cv2.imshow("img",img)
    cv2.imshow("h",h)
    cv2.imshow("s",s)
    cv2.imshow("v",v)
    '''
    cv2.imshow("l",l)
    cv2.imshow("u",u)
    cv2.imshow("v_",v_)
    cv2.imshow("L",L)
    cv2.imshow("A",A)
    cv2.imshow("B",B)
    cv2.imshow("H",H)
    cv2.imshow("S",S)
    cv2.imshow("L",L)
    cv2.imshow("Y",Y)
    cv2.imshow("Cr",Cr)
    cv2.imshow("Cb",Cb)  
    '''
    key = cv2.waitKey(0)
    if key == 27:
        cv2.destroyAllWindows()
if __name__ == "__main__":
    main()