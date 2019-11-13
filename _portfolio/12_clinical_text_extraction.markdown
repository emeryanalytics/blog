---
layout: post
title: Clinical Text Extraction
description: Scala, Spark, cTAKES
img: /img/thumbnail_clinical_text.jpg
---

The project aims to demonstrate the use of Natural Language Processing to search for relevant clinical information from doctors' clinical notes.
- Implement natural language processing pipeline with Scala and cTAKES, a library with both rule-based and machine learning techniques, to extract clinical information from unstructured medical text
- Develop big data pipeline with Scala and Spark to process Terrabytes of clinical data, extract clinical information and store them in Accumulo (HBase with cell level security)

<b>Sample clinical note:<b/>
<div>
	<img class="col" src="{{ site.baseurl }}/img/clinical_text.jpg" alt="" title="Clinical Text" border="1"/>
</div>

<br/>
<b>Sample clinical information extracted (<span style="color:red;">Red</span> indicates incorrect extraction):<b/>
<div>
	<img class="col" src="{{ site.baseurl }}/img/clinical_concep_extraction.jpg" alt="" title="Clinical Text" border="1"/>
</div>