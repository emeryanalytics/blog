---
layout: post
title: Dead/Alive Plant detection
description: Python, Tensorflow, OpenCV
tag: object-recogniton
img: /img/thumbnail_plant_detection.jpg
---

The project aims to automate the manual dead/alive plant counting process in a greenhouse. The manual process can takes 1 day for each experiment.
- Define data annotation protocol for scientists in order to obtain bounding box annotation for trays in a greenhouse bench, and dot annotation for dead/alive plants in a tray
- Develop deep learning segmentation model (U-net) to segment out individual trays in high resolution bench images using Python and Tensorflow
- Develop computer vision models (color threshold-based) to detect individual plants in a tray and classify them as dead/alive using Python, OpenCV
- Deploy the models in production, reducing the counting time from 1 day (manual counting) to 1 hour

<div>
	<img class="col" src="{{ site.baseurl }}/img/plant_detection.jpg" alt="" title="Dead/Alive Plant detection" border="1"/>
</div>