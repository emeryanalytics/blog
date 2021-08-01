---
layout: post
title: Road Object Detection and Tracking
description: AWS SageMaker, MLflow, Tensorflow, OpenCV, Python
tag: object-recognition
img: /img/thumbnail_road_object.jpg
---

Our operations team usually needs to manually go through millions of image frames, annotate road objects and extract their characteristics (size, type, text, position, etc). This project aims to detect and track road traffic signs and traffic signals in a sequence of image frames to facilitate road object extraction.
- Implement deep learning model training pipeline to detect different traffic signs and traffic signals using Python and Tensorflow on AWS SageMaker
- Integrage hyperparameter tuning and model selection into model training pipeline on AWS SageMaker and MLflow
- Implement object attribute extraction prototype  to extract attributes (type, text) for detected traffic signs/signals
- Implement object tracking algorithm with Python and OpenCV to track the same detected traffic signs/signals in multiple image frames
- Deploy the final solution on production, reducing the number of image frames that need manual reviewed by 50% while maintaining missed objects rate of less than 2%.

<div>
	<img class="col" src="{{ site.baseurl }}/img/road_object_overview.jpg" alt="" title="Road Object Overview" border="1"/><br/><br/>    
	<img class="col" src="{{ site.baseurl }}/img/road_attribute_extraction.jpg" alt="" title="Road Object Attribute Extraction" border="1"/><br/><br/>   
	<img class="col" src="{{ site.baseurl }}/img/road_object_tracking.gif" alt="" title="Road Object Tracking" border="1"/><br/><br/>   
</div>