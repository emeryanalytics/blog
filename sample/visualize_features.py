# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 12:44:08 2019

@author: linhb
"""

import os
import numpy as np
import cv2
import mahotas

def extract_hu_moments(img):
    """Extract Hu Moments feature of an image. Hu Moments are shape descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains 7 Hu Moments of the image
    """
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = cv2.HuMoments(cv2.moments(gray)).flatten()
    return feature

def extract_haralick(img):
    """Extract Haralick features of an image. Haralick features are texture descriptors.
    :param img: ndarray, BGR image
    :return feature: ndarray, contains 13 Haralick features of the image
    """

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    feature = mahotas.features.haralick(gray).mean(axis=0)
    return feature

path = 'C:/SourceCode/blog/img/'
file_list = os.listdir(path)

for filename in file_list:
    if 'letter' in filename.lower():
        img = cv2.imread(os.path.join(path, filename))
        hu = extract_hu_moments(img)
        hu = [np.round(x, 7) for x in hu]
        print(filename)
        print(hu)
        
for filename in file_list:
    if 'texture' in filename.lower():
        img = cv2.imread(os.path.join(path, filename))
        hu = extract_haralick(img)
        hu = [np.round(x, 2) for x in hu]
        print(filename)
        print(hu)
        
    