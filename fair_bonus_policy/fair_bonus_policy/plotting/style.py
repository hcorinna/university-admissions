#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 11 13:00:47 2019

@author: hertweck
"""

import matplotlib.pyplot as plt
#plt.style.use('fivethirtyeight')

import seaborn as sns
sns.set_style("whitegrid")

colors = [[86/255,180/255,233/255], [0,114/255,178/255], [230/255,159/255,0], [0,158/255,115/255], [213/255,94/255,0], [0,0,0]] 

palette1 = ['#eb4d4b', '#22a6b3']
palette2 = ['#f0932b', '#6ab04c']


def paper(plotter, params, xlabel, ylabel, filename=None, scale=1, height=1.2, figsize=(20,12)):
    plt.figure(figsize=figsize)
    ax = plotter(*params)
    ax.set_xlabel(xlabel,fontsize=40*scale)
    ax.set_ylabel(ylabel,fontsize=40*scale)
    ax.tick_params(labelsize=32*scale)
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, height),
          ncol=3, fancybox=False, shadow=False, prop={'size': 32*scale})
#    plt.setp(ax.get_legend().get_texts(), fontsize=28*scale) # for legend text
#    plt.setp(ax.get_legend().get_title(), fontsize=28*scale) # for legend title
    if filename is not None:
        plt.savefig(filename)
    plt.show()