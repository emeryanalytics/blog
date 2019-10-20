---
layout: post
title:  Tutorial: Skin lesion classification - Part 1
date:   2019-10-20 16:40:16
description: Classify skin lesion images as malignant/benign using global image descriptors
---

##Introduction##
In this tutorial, we are going to understand how to extract features from images build a model to classify the images into difference classes.<br/>
We will use a skin lesion dataset in this tutorial and we will look at different ways to extract features:
* Extract global image features with shape, texture and color descriptors (Part 1)
* Extract local image features with keypoint descriptors (Part 2)
* Learn relevant image features with convolutional neural network (Part 3)


##Global image features##
Before convolutional neural network architecture becomes popular, image processing relies on "hand-crafted" feature engineering to extract meaningful features from images.
Global features describe an image as a whole to the generalize the entire object, while  the local features describe smaller image patches.
Global features include contour representations, shape descriptors, texture and color features. In this tutorial, several common global features are used to train a skin lesion classification model.
* _Hu Moments_ is a shape descriptor that comprises 7 numbers calculated using central moments that are invariant to image transformations. The 7 moments are invariant to translation, scale, and rotation (only the 7th moment’s sign changes for image reflection).

| Image  | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
| ------ | -------- | -------- | ------------- |
| <img src="{{ site.baseurl }}/img/letter_K.png"/> | 0.0011752 | 0.0000005 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O.png"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_rotated.png"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_scaled.png"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_shifted.png"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |


* _Haralick features_: is a texture descriptor calculated from a gray level co-occurrence matrix (GLCM)
* _HSV Color histogram_: HSV (Hue, Saturation, Value) color space is closely corresponds to the human visual perception of color. To obtain HSV histogram, we devide hue scale, saturation scale, and intensity scale into 8 groups. By combining each of these groups, we get a total of 512 cells to represent a 512-component HSV color histogram. Then, the corresponding histogram component is determined by counting how many pixels belong to each group.

##Classification model##
Once we extract global features from images, we can use these features to train a model to classify malignant/benign skin lesion images.<br/>
We benchmark different base machine learning models from different classes of models:
* Logistic Regression
* K Nearest Neighbour, Support Vector Machine
* Random Forest and Gradient Boosting Tree).

We also implement model stacking that combines prediction from these base models to see if it helps improve model accuracy. The intuition of model stacking is that different models might perform better in some sections of feature space and perform worse in other sections. Model stacking would pay more attentions to models that perfrom better in certain sections of feature space.


