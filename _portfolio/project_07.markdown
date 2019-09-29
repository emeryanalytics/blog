---
layout: post
title: Drug Virtual Screening
description: Python, RDKit, PubChemPy, Scikit-learn
img: /img/chemistry.jpg
---

The project aims to evaluate the feasibility of using machine learning to predict efficacy of molecules against certain diseases based on the molecules' chemical structures.
- Extract 1D, 2D and 3D molecular descriptors and fingerprints of molecules using Python and RDKit
- Implement different machine learning models and perform model stacking to predict molecules' efficacy based on their structure with 80% accuracy using Python and Scikit-learn
- Develop a data retrieve pipeline to collect chemical information from PubChem and apply the model to screen for molecules with potential efficacy against certain diseases

<div>
	<img class="col" src="{{ site.baseurl }}/img/chemistry.jpg" alt="" title="Model Stacking" border="1"/>
</div>
<div>
	<img class="col" src="{{ site.baseurl }}/img/chemistry_accuracy.jpg" alt="" title="Model result" border="1"/>
</div>