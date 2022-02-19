#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:55:54 2019

@author: hertweck
"""

import numpy as np

def minimize_disparity_for_cutoff(min_cutoff_constraint, cutoff, disparity, bonus_values):
    constraint_indices = np.where(np.array(cutoff) >= min_cutoff_constraint)[0]
    possible_disparity_values = np.array(disparity)[constraint_indices]
    min_disparity_indices = locate_min(possible_disparity_values)
    indices = constraint_indices[min_disparity_indices]
    possible_bonus_values = np.array(bonus_values)[indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in possible_bonus_values])
    index = indices[min_bonus_index]
    ideal_bonus = bonus_values[index]
    min_disparity = disparity[index]
    return ideal_bonus, min_disparity

def minimize_disparity_for_utility(min_utility_constraint, utility, disparity, bonus_values):
    constraint_indices = np.where(np.array(utility) >= min_utility_constraint)[0]
    possible_disparity_values = np.array(disparity)[constraint_indices]
    min_disparity_indices = locate_min(possible_disparity_values)
    indices = constraint_indices[min_disparity_indices]
    possible_bonus_values = np.array(bonus_values)[indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in possible_bonus_values])
    index = indices[min_bonus_index]
    ideal_bonus = bonus_values[index]
    min_disparity = disparity[index]
    return ideal_bonus, min_disparity

def maximize_utility(max_disparity_constraint, utility, disparity, bonus_values):
    constraint_indices = np.where(np.array(disparity) <= max_disparity_constraint)[0]
    possible_utility_values = np.array(utility)[constraint_indices]
    max_utility_indices = locate_max(possible_utility_values)
    indices = constraint_indices[max_utility_indices]
    possible_bonus_values = np.array(bonus_values)[indices]
    min_bonus_index = np.argmin([abs(bonus) for bonus in possible_bonus_values])
    index = indices[min_bonus_index]
    ideal_bonus = bonus_values[index]
    max_utility = disparity[index]
    return ideal_bonus, max_utility

def locate_min(a):
    smallest = min(a)
    return [index for index, element in enumerate(a) if smallest == element]

def locate_max(a):
    largest = max(a)
    return [index for index, element in enumerate(a) if largest == element]