#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:52:12 2020

@author: hertweck
"""

import matplotlib.pyplot as plt
from fair_bonus_policy.clean_data import prepare_data
from fair_bonus_policy.measures import mm_measures

# =============================================================================
# Plot utility vs disparity for top programs
# =============================================================================

def utility_disparity(title, run_matching_algorithm, students, universities, TOP_K, bonus_values, sorted_schools, quality_of_school):
    results = calculate_disparity_utility(run_matching_algorithm, students, universities, TOP_K, bonus_values, sorted_schools, quality_of_school)
    plot_utility_disparity(title, results, TOP_K)
    #plot_utility_or_disparity(title, bonus_values, results, True, TOP_K)
    #plot_utility_or_disparity(title, bonus_values, results, False, TOP_K)

def calculate_disparity_utility(run_matching_algorithm, students, universities, TOP_K, bonus_values, sorted_schools, quality_of_school):
    results = {}
    top_schools = {}
    for k_schools in TOP_K:
        results[k_schools] = {'x_values': [], 'y_values': []}
        top_schools[k_schools] = set([int(xx) for xx, _ in sorted_schools[:k_schools]])

    for bonus in bonus_values:
        prepare_data.apply_bonus(students, bonus)
        students, universities = run_matching_algorithm(students, universities)
        for k_schools in TOP_K:
            disparity = mm_measures.admissions_disparity(universities, students, top_schools[k_schools], normed = True)
            utility = mm_measures.admissions_utility(universities, students, quality_of_school)

            results[k_schools]['x_values'].append(utility)
            results[k_schools]['y_values'].append(disparity)
    return results

def plot_utility_disparity(title, results, TOP_K):
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Utility')
    ax.set_ylabel('Disparity')
    for k_schools in TOP_K:
        ax.plot(results[k_schools]['x_values'], 
                   results[k_schools]['y_values'],
                   label = "top {} programs".format(k_schools))
    ax.legend()

def plot_utility_or_disparity(title, bonus_values, results, show_utility, TOP_K):
    if show_utility:
        ylabel = 'Utility'
        values = 'x_values'
    else:
        ylabel = 'Disparity'
        values = 'y_values'
        
    fig, ax = plt.subplots(1,1,figsize = (10,10));
    ax.set_title(title)
    ax.set_xlabel('Bonus values')
    ax.set_ylabel(ylabel)
    for k_schools in TOP_K:
        ax.plot(bonus_values, 
                   results[k_schools][values],
                   label = "top {} programs".format(k_schools))
    ax.legend()