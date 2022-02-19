#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:24:55 2019

@author: hertweck
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Colorblind-friendly colors
colors = [[0,0,0], [230/255,159/255,0], [86/255,180/255,233/255], [0,158/255,115/255],
          [213/255,94/255,0], [0,114/255,178/255]]

def plot_proportion_steps(proportion, steps, label, color, title, xlabel='Preference (1 is first choice, 2 is second choice etc.)', ylabel='Percentage'):
    for g in range(len(label)):
        plt.plot(steps, [x[g] for x in proportion.values()], label=label[g], color=color[g])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.show()
    
def plot_proportion_density(proportion, label, color, title, xlabel):
    for g in range(len(label)):
        sns.distplot([x[g] for x in proportion.values()], hist = False, kde = True, kde_kws = {'linewidth': 3}, label=label[g], color=color[g])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.show()
    
def plot_density(density, label, color, title, xlabel):
    for g in range(len(label)):
        sns.distplot(density[g], hist = False, kde = True, kde_kws = {'linewidth': 3}, label=label[g], color=color[g])
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel('Density')
    plt.show()
    
def compare_proportion_stacked(proportion, program_types, labels, colors):
    bottom = np.zeros(len(program_types))
    for g in range(len(labels)):
        height = [x[g] for x in proportion.values()]
        plt.bar(program_types, height, bottom=bottom, color=colors[g], label=labels[g])
        bottom = [bottom[i] + height[i] for i in program_types] 
    plt.xticks(program_types, range(1,len(program_types) + 1))
    plt.title('Share of subgroup in cluster')
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()
    
def compare_proportion_side_by_side(proportion, program_types, labels, colors, title):
    bar_width = 1 / (len(labels) + 1)
    for g in range(len(labels)):
        r = [x + g * bar_width for x in program_types]
        plt.bar(r, [x[g] for x in proportion.values()], width=bar_width, color=colors[g], label=labels[g])
    plt.xticks(program_types, range(1,8))
    plt.title(title)
    plt.xlabel('Cluster')
    plt.ylabel('Percentage')
    plt.legend()
    plt.show()
    
def plot_statistical_parity_measures(proportion, steps, sensitive_attribute, xlabel, base):
    title = ' with ' + sensitive_attribute + ' as sensitive attribute based on ' + base + ' applicants'
    spd = 'Statistical parity difference'
    di = 'Disparate impact'
    plt.plot(steps, [v[0] for v in proportion.values()], color=colors[1], alpha=0.5)
    if base!='all':    
        plt.axhline(y=-0.1, color=colors[1], linestyle='--')
    plt.title(spd + title)
    plt.xlabel(xlabel)
    plt.ylabel(spd)
    plt.show()
    plt.plot(steps, [v[1] for v in proportion.values()], color=colors[2], alpha=0.5)
    plt.axhline(y=0.8, color=colors[2], linestyle='--')
    plt.title(di + title)
    plt.xlabel(xlabel)
    plt.ylabel(di)
    plt.show()
    
def plot_statistical_parity_measures_scatter(proportion, steps, sensitive_attribute, xlabel, base):
    title = ' with ' + sensitive_attribute + ' as sensitive attribute based on ' + base + ' applicants'
    spd = 'Statistical parity difference'
    di = 'Disparate impact'
    plt.scatter(steps, [v[0] for v in proportion.values()], color=colors[1], alpha=0.5)
    if base!='all':    
        plt.axhline(y=-0.1, color=colors[1], linestyle='--')
        plt.axhline(y=0.1, color=colors[1], linestyle='--')
    plt.title(spd + title)
    plt.xlabel(xlabel)
    plt.ylabel(spd)
    plt.show()
    plt.scatter(steps, [v[1] for v in proportion.values()], color=colors[2], alpha=0.5)
    plt.axhline(y=0.8, color=colors[2], linestyle='--')
    plt.axhline(y=1.2, color=colors[2], linestyle='--')
    plt.title(di + title)
    plt.xlabel(xlabel)
    plt.ylabel(di)
    plt.show()
    
def plot_statistical_parity_measures_density(proportion, sensitive_attribute, base):
    title = ' with ' + sensitive_attribute + ' as sensitive attribute based on ' + base + ' applicants'
    spd = 'Statistical parity difference'
    di = 'Disparate impact'
    plt.hist([v[0] for v in proportion.values() if np.abs(v[0]) != float('inf')], bins=20, color=colors[1], alpha=0.5)
    if base!='all':
        plt.axvline(x=-0.1, color=colors[1], linestyle='--')
    plt.title(spd + title)
    plt.xlabel(spd)
    plt.ylabel('Frequency')
    plt.show()
    plt.hist([v[1] for v in proportion.values() if v[1] != float('inf')], bins=20, color=colors[2], alpha=0.5)
    plt.axvline(x=0.8, color=colors[2], linestyle='--')
    plt.title(di + title)
    plt.xlabel(di)
    plt.ylabel('Frequency')
    plt.show()

def plot_kendall_density(kendall, label_subgroups, color, label_measures, linestyle):
    for subgroup in range(len(label_subgroups)):
        for measure in range(len(label_measures)):
            sns.distplot(kendall[subgroup][measure], hist = False, kde = True, kde_kws = {'linewidth': 3, 'color': color[subgroup], 'label':label_subgroups[subgroup] + ' (' + label_measures[measure] + ')', 'linestyle':linestyle[measure], 'alpha':0.5})
    plt.title('Kendall')
    plt.xlabel('Kendall')
    plt.ylabel('Density')
    plt.show()
    
def plot_ndcg_density(kendall, label_subgroups, color, label_measures, linestyle):
    for subgroup in range(len(label_subgroups)):
        for measure in range(len(label_measures)):
            sns.distplot(kendall[subgroup][measure], hist = False, kde = True, kde_kws = {'linewidth': 3, 'color': color[subgroup], 'label':label_subgroups[subgroup] + ' (' + label_measures[measure] + ')', 'linestyle':linestyle[measure], 'alpha':0.5})
    plt.title('NDCG')
    plt.xlabel('NDCG')
    plt.ylabel('Frequency')
    plt.show()
    
def plot_ranking_distributions(distributions, labels, colors, number=3):
    """
    - distributions: Distributions of quality, risk and expected utility for different subgroups
    """
    for preference_index in range(number):
        plot_ranking_distribution(distributions, preference_index, labels, colors)
    
def plot_ranking_distribution(distributions, preference_index, labels, colors):
    """
    - distributions: Distributions of quality, risk and expected utility for different subgroups
    - preference_index: 0 = order by program quality, 1 = order by risk, 2 = order by expected utility
    """
    for d in range(len(distributions)):
        sns.distplot(distributions[d][preference_index], hist = False, kde = True, kde_kws = {'linewidth': 3, 'color':colors[d], 'label':labels[d], 'linestyle':measures_linestyle()[preference_index], 'alpha':0.5})
    plt.title(measures_label()[preference_index])
    plt.show()
    
def plot_choice_averages(choice_distributions, labels, colors):
    for preference_index in range(3):
        plot_choice_average(choice_distributions, preference_index, labels, colors)

def plot_choice_average(choice_distributions, preference_index, labels, colors):
    for d in range(len(choice_distributions)):
        plt.plot(range(1, len(choice_distributions[d][preference_index])+1), [np.mean(choice_qualities) for choice_qualities in choice_distributions[d][preference_index].values()], label=labels[d], color=colors[d])
    plt.title(measures_label()[preference_index] + ' by preference')
    plt.xlabel('Preference')
    plt.ylabel(measures_label()[preference_index])
    plt.legend()
    plt.show()
    
def setBoxColors(bp, c):
    single_counter = 0
    double_counter = [0,1]
    for color in c:
        plt.setp(bp['boxes'][single_counter], color=color)
        plt.setp(bp['caps'][double_counter[0]], color=color)
        plt.setp(bp['caps'][double_counter[1]], color=color)
        plt.setp(bp['whiskers'][double_counter[0]], color=color)
        plt.setp(bp['whiskers'][double_counter[1]], color=color)
        plt.setp(bp['fliers'][single_counter], markeredgecolor=color)
        plt.setp(bp['medians'][single_counter], color='black')
        single_counter += 1
        double_counter = [i+2 for i in double_counter]

def boxplot_comparison(data, group_labels, comparison_labels, c):
    # https://stackoverflow.com/a/16598291
    plt.figure(figsize=(20, 12))
    ax = plt.axes()

    positions = list(range(1,len(group_labels)+1))
    xticks = []
    for compare in data:
        # boxplot pair
        xticks.append(np.mean(positions))
        bp = plt.boxplot(compare, positions = positions, widths = 0.7)
        setBoxColors(bp, c)
        positions = [p + len(group_labels) + 1 for p in positions]

    # set axes limits and labels
    flattened_list = [item for sublist in data for subsublist in sublist for item in subsublist]
    difference = max(flattened_list) - min(flattened_list)
    plt.xlim(0, positions[0]-1)
    plt.ylim(min(flattened_list) - difference/10, max(flattened_list) + difference/10)
    plt.ylabel('Score',size=22)
    ax.set_xticklabels(comparison_labels)
    ax.set_xticks(xticks)
    plt.xticks(rotation=90, size=22)
    plt.yticks(size=22)
    
    # draw temporary red and blue lines and use them to create a legend
    lines = []
    for color in c:
        line, = plt.plot([1,1], color=color, linestyle='-')
        lines.append(line)
    plt.legend(lines,group_labels)
    for line in lines:
        line.set_visible(False)

    plt.show()
    
def plot_freqdist_freq(fd,
                       max_num=None,
                       cumulative=False,
                       title='Frequency plot',
                       linewidth=2):
    """
    As of NLTK version 3.2.1, FreqDist.plot() plots the counts 
    and has no kwarg for normalising to frequency. 
    Work this around here.
    source: https://martinapugliese.github.io/plotting-the-actual-frequencies-in-a-FreqDist-in-nltk/
    
    INPUT:
        - the FreqDist object
        - max_num: if specified, only plot up to this number of items 
          (they are already sorted descending by the FreqDist)
        - cumulative: bool (defaults to False)
        - title: the title to give the plot
        - linewidth: the width of line to use (defaults to 2)
    OUTPUT: plot the freq and return None.
    """

    tmp = fd.copy()
    norm = fd.N()
    for key in tmp.keys():
        tmp[key] = float(fd[key]) / norm
        print(key,':',tmp[key])

    if max_num:
        tmp.plot(max_num, cumulative=cumulative,
                 title=title, linewidth=linewidth)
    else:
        tmp.plot(cumulative=cumulative, 
                 title=title, 
                 linewidth=linewidth)

    return
    
def compare_distributions_in_different_plots(distributions, group_labels, comparison_labels, c):
    for i in range(len(comparison_labels)):
        data = [[distribution[i] for distribution in distributions]]
        boxplot_comparison(data, group_labels, [comparison_labels[i]], c)
    
def gender_labels():
    return ['Women', 'Men']

def gender_colors():
    return ['orange', 'cornflowerblue']

def ses_labels():
    return ['Low SES', 'Middle SES', 'High SES']

def ses_colors():
    return ['orange', 'cornflowerblue', 'green']

def income_labels():
    return ['Low-income', 'High-income']

def income_colors():
    return ['orange', 'green']

def measures_label():
    return ['Prestige','Risk','Sigmoid risk','Expected utility','Sigmoid expected utility']

def measures_linestyle():
    return ['-','--',':','-.',(0, (3,1,1,1,1,1))]