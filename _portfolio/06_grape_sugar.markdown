---
layout: post
title: Grape sugar estimation
description: Python, OpenCV, Scikit-learn
tag: object-recognition
img: /img/thumbnail_grape_sugar.jpg
---

Grape growers usually need to perform time-consuming lab test on sugar content to determine the best time for harvest. This project aims to replace sugar content lab test with a machine vision solution that comprises a multi-spectral imaging system and a machine learning model to estimate sugar content based on multi-spectral reflectance.
- Develop a multi-spectral imaging prototype with raspberry-pi, picamera and camera filters
- Collect multi-spectral images of grapes using the imaging prototype and the corresponding sugar content using a BRIX meter
- Pre-process and align channels of spectral images with traditional computer vision techniques (keypoint detection, template matching) using Python and OpenCV
- Develop regression-based machine learning models to estimate sugar content based on multi-spectral reflectance using Python and Scikit-learn

<div>
	<img class="col" src="{{ site.baseurl }}/img/grape_sugar.jpg" alt="" title="Multi-spectral imaging" border="1"/><br/><br/>
	<img class="col" src="{{ site.baseurl }}/img/grape_regression.jpg" alt="" title="Sugar regression model" border="1"/>
</div>