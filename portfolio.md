---
layout: page
title: Portfolio
permalink: /portfolio/
---

{%
list_or = []
list_gv = []
list_sp = []
list_cb = []
list_nlp = []
list_sa = []

for project in site.portfolio
    if 'object-recognition' in project.tag:
        list_or << project
    endif
    if 'geometric-vision' in project.tag:
        list_gv << project
    endif
    if 'signal-processing' in project.tag:
        list_sp << project
    endif
    if 'computational-biochemistry' in project.tag:
        list_cb << project
    endif
    if  'statistical-analysis' in project.tag:
        list_sa << project
    endif
    if 'natural-language-processing' in project.tag:
        list_nlp << project
    endif
endfor
        
list_projects = [list_or, list_gv, list_sp, list_cb, list_nlp] 
list_categories = ['Object Detection & Semantic Segmentation', 'Stereo Vision & 3D Reconstruction', 'Signal Processing', 'Computational Biochemistry', 'Statistical-Analysis', 'Natural Language Processing']

for i in 0..(length(list_projects))
    category = list_categories[i]
    projects = list_projects[i]
    for project in projects
%}

<h2>{{ category }}</h2>
{% if project.redirect %}
<div class="project">
    <div class="thumbnail">
        <a href="{{ project.redirect }}" target="_blank">
        {% if project.img %}
        <img class="thumbnail" src="{{ site.baseurl }}/{{ project.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <p>{{ project.description }}</p>
        </span>
        </a>
    </div>
</div>
{% else %}

<div class="project ">
    <div class="thumbnail">
        <a href="{{ site.baseurl }}{{ project.url }}">
        {% if project.img %}
        <img class="thumbnail" src="{{ site.baseurl }}/{{ project.img }}"/>
        {% else %}
        <div class="thumbnail blankbox"></div>
        {% endif %}    
        <span>
            <h1>{{ project.title }}</h1>
            <br/>
            <p>{{ project.description }}</p>
        </span>
        </a>
    </div>
</div>

{% endif %}

{% 
    endfor
endfor %}
