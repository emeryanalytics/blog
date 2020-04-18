---
layout: post
title: Worm counting
description: Python, Tensorflow
tag: object-recogniton
img: /img/thumbnail_worm_counting.jpg
---

The project aims to automate the manual worm counting process which is tedious and time consuming for scientists. Each experiment can take 2 days for scientists to count egg/dead/alive worms.
- Develop machine vision protocol for scientists to collect worm images during their experiments
- Collect data annotation from the scientists and through Amazon MTurk
- Develop and evaluate deep learning models (Faster R-CNN, U-Net) to detect different worm types (egg, dead, alive) using Python and Tensorflow
- Deploy the best model on production, reducing the worm counting time from 2 days (manual counting) to 1 hour (automated counting with computer vision).

<div>
	<img class="col caption" src="{{ site.baseurl }}/img/worm_counting.jpg" alt="" title="Worm Counting" border="1"/>
</div>