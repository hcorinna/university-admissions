#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:44:43 2020

@author: hertweck
"""

import numpy as np
import matplotlib.pyplot as plt
from fair_bonus_policy.plotting import style
from fair_bonus_policy.utils import minimum
from fair_bonus_policy.measures import mm_measures


def plot_utility(title, bonus_values, utility):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel('Utility')
    ax.plot(bonus_values, utility)
    plt.axvline(x=0, label='Without bonus', color='black')
    max_utility_indices = minimum.locate_max(utility)
    bonuses_highest_utility = np.array(bonus_values)[max_utility_indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in bonuses_highest_utility])
    bonus_highest_utility = bonuses_highest_utility[min_bonus_index]
    plt.axvline(x=bonus_highest_utility, label='Bonus for maximum utility (' + str(bonus_highest_utility) + ')', color='darkorange')
    ax.legend()
    plt.show()
    
def plot_disparity(title, bonus_values, disparity):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel('Disparity')
    ax.plot(bonus_values, disparity)
    plt.axvline(x=0, label='Without bonus', color='black')
    min_disparity_indices = minimum.locate_min(disparity)
    bonuses_lowest_disparity = np.array(bonus_values)[min_disparity_indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in bonuses_lowest_disparity])
    bonus_lowest_disparity = bonuses_lowest_disparity[min_bonus_index]
    plt.axvline(x=bonus_lowest_disparity, label='Bonus for minimum disparity (' + str(bonus_lowest_disparity) + ')', color='darkorange')
    ax.legend()
    plt.show()
    
def plot_cutoff(title, bonus_values, cutoff):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel('Cutoff')
    ax.plot(bonus_values, cutoff)
    plt.axvline(x=0, label='Without bonus', color='black')
    max_cutoff_indices = minimum.locate_max(cutoff)
    bonuses_highest_cutoff = np.array(bonus_values)[max_cutoff_indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in bonuses_highest_cutoff])
    bonus_highest_cutoff = bonuses_highest_cutoff[min_bonus_index]
    plt.axvline(x=bonus_highest_cutoff, label='Bonus for maxium cutoff (' + str(bonus_highest_cutoff) + ')', color='darkorange')
    ax.legend()
    plt.show()
    
def plot_statistical_parity_difference(title, bonus_values, statistical_parity_difference):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel('Statistical disparity difference')
    ax.plot(bonus_values, statistical_parity_difference)
    plt.axvline(x=0, label='Without bonus', color='black')
    plt.axhline(y=0, label='Optimal statistical disparity difference', color=style.colors[1], linestyle='--')
    plt.axhline(y=-.1, label='Fairness boundary', color=style.colors[1], linestyle='--', alpha=0.5)
    plt.axhline(y=.1, color=style.colors[1], linestyle='--', alpha=0.5)
    min_disparity_indices = minimum.locate_min([abs(spd) for spd in statistical_parity_difference])
    bonuses_lowest_disparity = np.array(bonus_values)[min_disparity_indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in bonuses_lowest_disparity])
    bonus_lowest_disparity = bonuses_lowest_disparity[min_bonus_index]
    plt.axvline(x=bonus_lowest_disparity, label='Bonus for minimum statistical disparity difference (' + str(bonus_lowest_disparity) + ')', color='darkorange')
    ax.legend()
    plt.show() 

def plot_disparate_impact(title, bonus_values, disparate_impact):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel('Disparate impact')
    ax.plot(bonus_values, disparate_impact)
    plt.axvline(x=0, label='Without bonus', color='black')
    plt.axhline(y=1, label='Optimal statistical disparity difference', color=style.colors[2], linestyle='--')
    plt.axhline(y=0.8, label='Fairness boundary', color=style.colors[2], linestyle='--', alpha=0.5)
    plt.axhline(y=1.2, color=style.colors[2], linestyle='--', alpha=0.5)
    min_disparity_indices = minimum.locate_min([abs(di - 1) for di in disparate_impact])
    bonuses_lowest_disparity = np.array(bonus_values)[min_disparity_indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in bonuses_lowest_disparity])
    bonus_lowest_disparity = bonuses_lowest_disparity[min_bonus_index]
    plt.axvline(x=bonus_lowest_disparity, label='Bonus for minimum disparate impact (' + str(bonus_lowest_disparity) + ')', color='darkorange')
    ax.legend()
    plt.show()
    
def plot_objective_function_broken_axis(title, bonus_values, utility, statistical_parity, _lambda):
    # broken axis plot: https://matplotlib.org/examples/pylab_examples/broken_axis.html
    fig, (ax, ax2) = plt.subplots(2, 1, sharex=True,figsize = (10,10))
    ax.set_title(title, size=24)
    ax2.set_xlabel('Bonus values', size=22)
    ax2.set_ylabel('Function values', size=22)
    ax.spines['bottom'].set_visible(False)
    ax2.spines['top'].set_visible(False)
    ax.xaxis.tick_top()
    ax.tick_params(labeltop='off')  # don't put tick labels at the top
    ax2.xaxis.tick_bottom()
    
    utility_y = - np.array(utility)
    disparity_y = abs(np.array(statistical_parity)) * _lambda
    objective_function = utility_y + disparity_y
    
    ax.set_ylim(0, max(disparity_y) + 1)
    ax2.set_ylim(min(utility_y) - 20, max(utility_y) + 20)

    ax.plot(bonus_values, disparity_y, label=r'statistical_parity x $\lambda$', color=style.colors[1], linestyle='--')
    ax.plot(bonus_values, utility_y, label='- utility', color=style.colors[3], linestyle='--')
    ax.plot(bonus_values, objective_function, label='objective function')
    ax2.plot(bonus_values, disparity_y, label=r'statistical_parity x $\lambda$', color=style.colors[1], linestyle='--')
    ax2.plot(bonus_values, utility_y, label='- utility', color=style.colors[3], linestyle='--')
    ax2.plot(bonus_values, objective_function, label='objective function')
    
    d = .015  # how big to make the diagonal lines in axes coordinates
    # arguments to pass to plot, just so we don't keep repeating them
    kwargs = dict(transform=ax.transAxes, color='k', clip_on=False)
    ax.plot((-d, +d), (-d, +d), **kwargs)        # top-left diagonal
    ax.plot((1 - d, 1 + d), (-d, +d), **kwargs)  # top-right diagonal
    
    kwargs.update(transform=ax2.transAxes)  # switch to the bottom axes
    ax2.plot((-d, +d), (1 - d, 1 + d), **kwargs)  # bottom-left diagonal
    ax2.plot((1 - d, 1 + d), (1 - d, 1 + d), **kwargs)  # bottom-right diagonal

    bonus_lowest_disparity = minimum.find_min_value_bonus(bonus_values, disparity_y)
    ax.axvline(x=bonus_lowest_disparity, linestyle=':', label='Bonus for minimum statistical disparity (' + str(bonus_lowest_disparity) + ')', color=style.colors[1])
    ax2.axvline(x=bonus_lowest_disparity, linestyle=':', label='Bonus for minimum statistical disparity (' + str(bonus_lowest_disparity) + ')', color=style.colors[1])
    bonus_lowest_utility = minimum.find_min_value_bonus(bonus_values, utility_y)
    ax.axvline(x=bonus_lowest_utility, linestyle=':', label='Bonus for minimum utility (' + str(bonus_lowest_utility) + ')', color=style.colors[3])
    ax2.axvline(x=bonus_lowest_utility, linestyle=':', label='Bonus for minimum utility (' + str(bonus_lowest_utility) + ')', color=style.colors[3])
    bonus_lowest_objective = minimum.find_min_value_bonus(bonus_values, objective_function)
    ax.axvline(x=bonus_lowest_objective, linestyle=':', label='Bonus for minimum objective function (' + str(bonus_lowest_objective) + ')')
    ax2.axvline(x=bonus_lowest_objective, linestyle=':', label='Bonus for minimum objective function (' + str(bonus_lowest_objective) + ')')
    
    ax.legend()
    plt.show()
    
def plot_objective_function(title, bonus_values, utility, statistical_parity, _lambda):
    fig, ax = plt.subplots(1, 1, figsize = (10,10))
    ax.set_title(title, size=24)
    ax.set_xlabel('Bonus values', size=22)
    ax.set_ylabel('Function values', size=22)
    
    y, utility_difference_y, disparity_y = mm_measures.objective_function(bonus_values, utility, statistical_parity, _lambda)
    
    ax.set_ylim(-.5, max(disparity_y) + 1)

    ax.plot(bonus_values, disparity_y, label=r'statistical_parity x $\lambda$', color=style.colors[1], linestyle='--')
    ax.plot(bonus_values, utility_difference_y, label='utility loss', color=style.colors[3], linestyle='--')
    ax.plot(bonus_values, y, label='objective function')

    bonus_lowest_disparity = minimum.find_min_value_bonus(bonus_values, disparity_y)
    ax.axvline(x=bonus_lowest_disparity, linestyle=':', label='Bonus for minimum statistical disparity (' + str(bonus_lowest_disparity) + ')', color=style.colors[1])
    bonus_lowest_utility_difference = minimum.find_min_value_bonus(bonus_values, utility_difference_y)
    ax.axvline(x=bonus_lowest_utility_difference, linestyle=':', label='Bonus for minimum utility difference (' + str(bonus_lowest_utility_difference) + ')', color=style.colors[3])
    bonus_lowest_objective = minimum.find_min_value_bonus(bonus_values, y)
    ax.axvline(x=bonus_lowest_objective, linestyle=':', label='Bonus for minimum objective function (' + str(bonus_lowest_objective) + ')')
    
    ax.legend()
    plt.show()
    for bonus, y_value in zip(bonus_values, y):
        print(bonus, ':', y_value)