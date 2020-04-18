---
layout: post
title: Skin Lesion Classification
description: Python, OpenCV, Scitki-learn, Keras
tag: object-recognition
img: /img/thumbnail_skin_lesion.jpg
---

The project aims to demonstrate the use of Computer Vision to assist diagnosis by classifying whether a skin lesion is malignant vs benign.
- Develop image processing pipeline to extract global features (shape, texture, color descriptors) and local features (visual bag of words with keypoint descriptors) of skin lesion images 
- Train machine learning models and implemented model stacking for skin lesion image classification
- Implement convolutional neural networks and transfer learning to improve the classification accuracy

<div>
	<img class="col" src="{{ site.baseurl }}/img/skin_lesion_stacking_model.jpg" alt="" title="Stacking Model" border="1"/>
</div>

<div>
	<img class="col" src="{{ site.baseurl }}/img/skin_lesion_cnn_model.jpg" alt="" title="Convolutional Neural Network" border="1"/>
</div>