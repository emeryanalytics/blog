---
layout: post
title: Cardiac MRI Segmentation
description: Python, Pydicom, Keras, OpenCV
tag: object-recognition
img: /img/thumbnail_cardiac_segmentation.jpg
---

The project aims to demonstrate the use of Computer Vision to assist diagnosis by segmenting endocardium of left ventricle, which is responsible for pumping oxygenated blood to tissues in the body.
- Develop image processing pipeline to parse cardiac DICOM images and extract out pixel data using Python and Pydicom
- Implement a deep convolutional neural network (U-Net), together with Dice loss/Jaccard loss, to perform endocardial segmentation on 2D MRI slice using Python and Keras
- Perform post-processing with contour analysis on the model's prediction to improve segmentation quality using Python and OpenCV

<div>
	<img class="col" src="{{ site.baseurl }}/img/u_net.jpg" alt="" title="U Net" border="1"/>
</div>
