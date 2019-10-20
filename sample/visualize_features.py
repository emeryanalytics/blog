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

path = 'D:/Training/Source_Code/Other/blog/img/'
file_list = os.listdir(path)

for filename in file_list:
    if 'letter' in filename.lower():
        img = cv2.imread(os.path.join(path, filename))
        hu = extract_hu_moments(img)
        hu = [np.round(x, 7) for x in hu]
        print(filename)
        print(hu)
        
    