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
| <img src="{{ site.baseurl }}/img/letter_K.png" width="40"/> | 0.0011752 | 0.0000005 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| <img src="{{ site.baseurl }}/img/letter_O.png" width="40"/> | 0.0012954 | 0.0000004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| <img src="{{ site.baseurl }}/img/letter_O_rotated.png" width="40"/> | 0.0012954 | 0.0000004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| <img src="{{ site.baseurl }}/img/letter_O_shifted.png" width="40"/> | 0.0012954 | 0.0000004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |
| <img src="{{ site.baseurl }}/img/letter_O_scaled.png" width="40"/> | 0.0012954 | 0.0000004 | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

<br/>
* _Haralick features_: is a texture descriptor calculated from a gray level co-occurrence matrix (GLCM). Haralick features are rotational and translational invariant, but not scale invariant.<br/>

| Image  | H1 | H2 | H3 | H4 | H5 | H6 | H7 | H8 | H8 | H10 | H11 | H12 | H13 |
| ------ | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| <img src="{{ site.baseurl }}/img/texture_2.jpg" width="40"/> | 0.0 | 539.46 | 0.83 | 1611.63 | 0.09 | 175.46 | 5907.08 | 8.18 | 13.23 | 0.0 | 5.51 | -0.16 | 0.95 |
| <img src="{{ site.baseurl }}/img/texture_1.jpg" width="40"/> | 0.19 | 677.4 | 0.89 | 3065.19 | 0.5 | 71.28 | 11583.34 | 5.48 | 8.02 | 0.0 | 4.13 | -0.37 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_rotated.jpg" width="40"/> | 0.18 | 681.22 | 0.89 | 3066.15 | 0.5 | 71.3 | 11583.36 | 5.55 | 8.12 | 0.0 | 4.16 | -0.37 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_shifted.jpg" width="40"/> | 0.17 | 711.63 | 0.88 | 3071.9 | 0.48 | 71.58 | 11575.98 | 5.7 | 8.35 | 0.0 | 4.26 | -0.35 | 0.99 |
| <img src="{{ site.baseurl }}/img/texture_1_Scaled.jpg" width="40"/> | 0.7 | 477.84 | 0.76 | 1001.88 | 0.85 | 17.94 | 3529.7 | 1.93 | 2.55 | 0.0 | 1.62 | -0.5 | 0.91 |

<br/>
* _HSV Color histogram_: HSV (Hue, Saturation, Value) color space is closely corresponds to the human visual perception of color. To obtain HSV histogram, we devide hue scale, saturation scale, and intensity scale into 8 groups. By combining each of these groups, we get a total of 512 cells to represent a 512-component HSV color histogram. Then, the corresponding histogram component is determined and normalized by counting how many pixels belong to each group. Since HSV Color histogram is based on pixel count, it is rotational, translational and scale invariant. 

### Classification models
Once we extract global features from images, we can use these features to train a model to classify malignant/benign skin lesion images. Since most of these features are rotational, translational and scale invariant, we don't have to worry much about images of different rotation, translation and scale.<br/>
We benchmark different base machine learning models from different classes of models:
* Logistic Regression
* K Nearest Neighbour
* Support Vector Machine
* Tree-based models (Random Forest, Gradient Boosting Tree)

We also implement model stacking that combines prediction from these base models to see if it helps improve model accuracy. The intuition of model stacking is that different models might perform better in some sections of feature space and perform worse in other sections. Model stacking would pay more attentions to models that perfrom better in certain sections of feature space.<br/>
<img src="{{ site.baseurl }}/img/skin_lesion_stacking_model.jpg" alt="" width="100%">

<br/>
### Code
Full code for <a href="https://github.com/liambll/skin-lesion-classification/blob/master/models/feature_extraction.py">image feature extraction</a> and <a href="https://github.com/liambll/skin-lesion-classification/blob/master/models/stacking_model.py">classification models</a>.

* We first implement functions to extract different image features:

```python
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
    
    
    def extract_color_histogram(img, n_bins=8):
        """Extract Color histogram of an image.
        :param img: ndarray, BGR image
        :return feature: ndarray, contains n_bins*n_bins*n_bins HSV histogram features of the image
        """
        
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV) # convert the image to HSV color-space
        hist  = cv2.calcHist([hsv], [0, 1, 2], None, [n_bins, n_bins, n_bins], [0, 180, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        feature = hist.flatten()
        return feature
```

* We read in images and extract features. Remember to normalize features because feature scale/normalization is important for some machine learning algorithms:
```python
    # Normalize features
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler(feature_range=(0, 1))
    X_train = scaler.fit_transform(X_train_all_features)
    X_test = scaler.transform(X_test_all_features)
```

* We can now train base machine learning models:
```python
    ########################
    # DEFINE BASE MODELS ###
    ########################
    models = []
    model_params = []
    model_names = []
    
    # Random forest model
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            models.append(RandomForestClassifier(max_features='sqrt', class_weight='balanced', random_state=0))
            model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth]})
            model_names.append('Random Forest')   
    
    # Boosted Tree
    for n_estimators in [500, 1000, 2000]:
        for max_depth in [3, 5, 7]:
            for learning_rate in [0.01, 0.1]:
                models.append(GradientBoostingClassifier(subsample=0.7, max_features='sqrt', random_state=0))
                model_params.append({'n_estimators':[n_estimators], 'max_depth':[max_depth], 'learning_rate':[learning_rate]})
                model_names.append('Gradient Boosting Machine')
    
    # SVM
    for kernel in ['linear', 'rbf']:
        for C in [1.0, 10.0, 100.0, 1000.0]:
            models.append(SVC(probability=True, gamma='auto', tol=0.001, cache_size=200, class_weight='balanced',
                              random_state=0,
                              decision_function_shape='ovr'))
            model_params.append({'kernel':[kernel], 'C':[C]})
            model_names.append('Support Vector Machine')
    
    # Logistic regression model
    for penalty in ['l1', 'l2']:
        for C in [1.0, 10.0, 100.0, 1000.0]:
            models.append(linear_model.LogisticRegression(max_iter=500, solver='liblinear', multi_class='ovr',
                                                          class_weight='balanced', random_state=0))
            model_params.append({'penalty':[penalty], 'C':[C]})
            model_names.append('Logistic Regression')
        
    # KNN
    for n_neighbors in [5, 10, 15]:
        for weights in ['uniform', 'distance']:
            models.append(KNeighborsClassifier())
            model_params.append({'n_neighbors':[n_neighbors], 'weights':[weights]})
            model_names.append('K Nearest Neighbour')
            
    ##################################
    # TRAIN AND EVALUATE BASE MODELS #
    ##################################
    fitted_models = []
    model_scores = []
    for i in range(len(models)):
        print('Evaluating model {} of {}: {}'.format((i+1), len(models), model_names[i]))
        model = models[i]
        fitted_cv, _, _ = train_model(model=model, X_train=X_train, y_train=y_train, parameters=model_params[i])
        fitted_whole_set, _, score = evaluate_model(model=fitted_cv, X_train=X_train, y_train=y_train,
                                                    X_test=X_test, y_test=y_test)
        fitted_models.append(fitted_whole_set)
        model_scores.append(score)
        print(model_names[i], score)
```

* If we want to try model stacking, we can take prediction from base models and add them to the feature space and train a model with new feature space.
```python          
    ###############################
    ### PREPARE DATA FOR STACKING #
    ###############################
    print('Preparing data for model stacking')
    from sklearn.preprocessing import OneHotEncoder
    label_encoder = OneHotEncoder(categories='auto', sparse=False)
    label_encoder.fit(np.unique(y_train).reshape(-1, 1))
    nb_classes = len(label_encoder.categories_[0])
    
    # Get base models' prediction for test set: simply use the trained models to predict on test set
    X_test_stack = np.zeros([X_test.shape[0], len(base_models)*nb_classes])
    for i in range(len(base_models)):
        model = base_models[i]
        X_test_stack[:, i*nb_classes:(i+1)*nb_classes] = label_encoder.transform(model.predict(X_test).reshape(-1, 1))
            
    # Get base models' prediction for train set: use 3-fold split, train model on 2 parts and predict on 3rd part
    splits = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=0).split(X=X_train, y=y_train)
    X_train_stack = np.zeros([X_train.shape[0], len(base_models)*nb_classes])
    for train_index, val_index in splits:
        # train and validation set
        X_tr, X_val = X_train[train_index], X_train[val_index]
        y_tr, _ = y_train[train_index], y_train[val_index]

        # Fit model
        for i in range(len(base_models)):
            model = base_models[i]
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')  # disable the warning on default optimizer
                model.fit(X_tr, y_tr)
            X_train_stack[val_index, i*nb_classes:(i+1)*nb_classes] = \
                label_encoder.transform(model.predict(X_val).reshape(-1, 1))

    # Add base models' predictions into the feature space
    X_train_stack = np.concatenate([X_train, X_train_stack], axis=-1)
    X_test_stack = np.concatenate([X_test, X_test_stack], axis=-1)
          
    ########################
    # DEFINE STACK MODELS ##
    ########################
    stack_models = []
    stack_model_names = []
    stack_model_params = []
    
    stack_models.append(linear_model.LogisticRegression(max_iter=500, solver='liblinear', multi_class='ovr',
                                                        class_weight='balanced', random_state=0))
    stack_model_names.append('Stack Logistic Regression')
    stack_model_params.append({'penalty':['l1', 'l2'], 'C':[1.0, 10.0, 100.0, 1000.0]})
    
    stack_models.append(RandomForestClassifier(class_weight='balanced', random_state=0))
    stack_model_names.append('Stack Random Forest')
    stack_model_params.append({'n_estimators':[500, 1000, 2000], 'max_depth':[3, 5, 7]})
    
    stack_models.append(GradientBoostingClassifier(subsample=0.7, max_features='sqrt', learning_rate=0.01,
                                                   random_state=0))
    stack_model_names.append('Stack Gradient Boosting Machine')
    stack_model_params.append({'n_estimators':[500, 1000, 2000], 'max_depth':[3, 5, 7]})
    
    stack_models.append(SVC(probability=True, gamma='auto', tol=0.001, cache_size=200, random_state=0,
                             decision_function_shape='ovr', class_weight='balanced'))
    stack_model_names.append('Stack Support Vector Machine')
    stack_model_params.append({'kernel':['linear', 'rbf'], 'C':[1.0, 10.0, 100.0, 1000.0]})
    
    stack_models.append(KNeighborsClassifier())
    stack_model_names.append('Stack K Nearest Neighbour')
    stack_model_params.append({'n_neighbors':[5, 10, 15], 'weights':['uniform', 'distance']})          

    #########################
    # EVALUATE STACK MODELS #
    #########################
    stack_fitted_models = []
    stack_model_scores = []
    for i in range(len(stack_models)):
        print('Evaluating model {} of {}: {}'.format((i+1), len(stack_models), stack_model_names[i]))
        model = stack_models[i]
        fitted_cv, _, _ = train_model(model=model, X_train=X_train_stack, y_train=y_train,
                                      parameters=stack_model_params[i])
        fitted_whole_set, _, score = evaluate_model(model=fitted_cv, X_train=X_train_stack, y_train=y_train,
                                                    X_test=X_test_stack, y_test=y_test)
        stack_fitted_models.append(fitted_whole_set)
        stack_model_scores.append(score)
        print(stack_model_names[i], score)
```



