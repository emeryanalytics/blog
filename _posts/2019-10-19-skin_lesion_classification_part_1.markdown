---
layout: post
title:  Tutorial - Skin lesion classification (Part 1)
date:   2019-10-19 16:40:16
description: Classify skin lesion images as malignant/benign using global image descriptors
---

### Introduction
In this tutorial, we are going to understand how to extract features from images build a model to classify the images into difference classes.<br/>
We will use a skin lesion dataset in this tutorial and we will look at different ways to extract features:
* Extract global image features with shape, texture and color descriptors (Part 1)
* Extract local image features with keypoint descriptors (Part 2)
* Learn relevant image features with convolutional neural network (Part 3)


### Global image features
Before convolutional neural network architecture becomes popular, image processing relies on "hand-crafted" feature engineering to extract meaningful features from images.
Global features describe an image as a whole to the generalize the entire object, while  the local features describe smaller image patches.
Global features include contour representations, shape descriptors, texture and color features. In this tutorial, several common global features are used to train a skin lesion classification model.
* _Hu Moments_ is a shape descriptor that comprises 7 numbers calculated using central moments that are invariant to image transformations. The 7 moments are invariant to translation, scale, and rotation (only the 7th moment’s sign changes for image reflection).<br/>

| Image  | H1 | H2 | H3 | H4 | H5 | H6 | H7 |
| ------ | --- | --- | --- | --- | --- | --- | --- |
| <img src="{{ site.baseurl }}/img/letter_K.png" width="30"/> | 0.0011752 | 0.0000005 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O.png" width="30"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_rotated.png" width="30"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_shifted.png" width="30"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |
| <img src="{{ site.baseurl }}/img/letter_O_scaled.png" width="30"/> | 0.0012954 | 0.0000004 | 0 | 0 | 0 | 0 | 0 |


* _Haralick features_: is a texture descriptor calculated from a gray level co-occurrence matrix (GLCM). Haralick features are rotational and translational invariant, but not scale invariant.<br/>

| Image  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H8 | H10 | H11 | H12 | H13 |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <img src="{{ site.baseurl }}/img/texture_2.jpg" width="30"/> | 0.0 | 539.46 | 0.83 | 1611.63 | 0.09 | 175.46 | 5907.08 | 8.18 | 13.23 | 0.0 | 5.51 | -0.16 | 0.95 |
| <img src="{{ site.baseurl }}/img/texture_1.jpg" width="30"/> | 0.19 | 677.4 | 0.89 | 3065.19 | 0.5 | 71.28 | 11583.34 | 5.48 | 8.02 | 0.0 | 4.13 | -0.37 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_rotated.jpg" width="30"/> | 0.18 | 681.22 | 0.89 | 3066.15 | 0.5 | 71.3 | 11583.36 | 5.55 | 8.12 | 0.0 | 4.16 | -0.37 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_shifted.jpg" width="30"/> | 0.17 | 711.63 | 0.88 | 3071.9 | 0.48 | 71.58 | 11575.98 | 5.7 | 8.35 | 0.0 | 4.26 | -0.35 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_Scaled.jpg" width="30"/> | 0.7 | 477.84 | 0.76 | 1001.88 | 0.85 | 17.94 | 3529.7 | 1.93 | 2.55 | 0.0 | 1.62 | -0.5 | 0.91 |


* _HSV Color histogram_: HSV (Hue, Saturation, Value) color space is closely corresponds to the human visual perception of color. To obtain HSV histogram, we devide hue scale, saturation scale, and intensity scale into 8 groups. By combining each of these groups, we get a total of 512 cells to represent a 512-component HSV color histogram. Then, the corresponding histogram component is determined and normalized by counting how many pixels belong to each group. Since HSV Color histogram is based on pixel count, it is rotational, translational and scale invariant. 

<br/>
### Classification models
Once we extract global features from images, we can use these features to train a model to classify malignant/benign skin lesion images. Since most of these features are rotational, translational and scale invariant, we don't have to worry much about images of different rotation, translation and scale.<br/>
We benchmark different base machine learning models from different classes of models:
* Logistic Regression
* K Nearest Neighbour
* Support Vector Machine
* Tree-based models (Random Forest, Gradient Boosting Tree)

We also implement model stacking that combines prediction from these base models to see if it helps improve model accuracy. The intuition of model stacking is that different models might perform better in some sections of feature space and perform worse in other sections. Model stacking would pay more attentions to models that perfrom better in certain sections of feature space.<br/>
<img src="{{ site.baseurl }}/img/skin_lesion_stacking_model.jpg" alt="" width="80%"><br/>

