---
layout: post
title: Road Object Triangulation
description: OpenCV, Python
tag: geometric-vision
img: /img/thumbnail_road_object_triangulation.jpg
---

Our operations team usually go through a sequence of image frames, annotate road objects and identify the same object in multiple frames. Once that is done, we need to have an accurate algorithm to estimate an object's real world GPS coordinates.
- Improve data processing pipeline to process Vehicle's position and orientation IMU data, read in camera intrinsic and extrinsic parameters, and perform triangulation to estimate an object's real world GPS coordinates using two image frames, reducing the estimation error to less than 1 meter.
- Implement bundle adjustment algorithm to estimate an object's real world GPS coordinates using more than two image frames.

<div>
	<img class="col" src="{{ site.baseurl }}/img/road_object_triangulation.jpg" alt="" title="Road Object Triangulation" border="1"/>      
</div>