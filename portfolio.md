---
layout: page
title: Portfolio
permalink: /portfolio/
---

{% assign list_projects = "" | split: ',' %}
{% assign list_categories = "Object Detection & Semantic Segmentation, Stereo Vision & 3D Reconstruction, Signal Processing, Computational Biochemistry, Statistical Analysis, Natural Language Processing" | split: ',' %}

{% assign list_or = site.portfolio | 
      where_exp: "project", "project.tag contains 'object-recognition'" %}
{% assign list_projects = list_projects | push: list_or %}

{% assign list_gv = site.portfolio | 
      where_exp: "project", "project.tag contains 'geometric-vision'" %}
{% assign list_projects = list_projects | push: list_gv %}

{% assign list_sp = site.portfolio | 
      where_exp: "project", "project.tag contains 'signal-processing'" %}
{% assign list_projects = list_projects | push: list_sp %}

{% assign list_cb = site.portfolio | 
      where_exp: "project", "project.tag contains 'computational-biochemistry'" %}
{% assign list_projects = list_projects | push: list_cb %}

{% assign list_sa = site.portfolio | 
      where_exp: "project", "project.tag contains 'statistical-analysis'" %}
{% assign list_projects = list_projects | push: list_sa %}

{% assign list_nlp = site.portfolio | 
      where_exp: "project", "project.tag contains 'natural-language-processing'" %}
{% assign list_projects = list_projects | push: list_nlp %}

{% assign nb_categories = list_categories.size | minus: 1 %}

{% for i in (0..nb_categories) %}
{% assign category = list_categories[i] %}
{% assign projects = list_projects[i] %}

<hr class="clearboth"/>
<h4 class="clearboth">{{ category }}</h4>
{% for project in projects %}
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
{% endfor %}

<br class="clearboth"/>
{% endfor %}
