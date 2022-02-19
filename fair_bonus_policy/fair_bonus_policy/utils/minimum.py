#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 12 15:47:34 2020

@author: hertweck
"""

import numpy as np

def find_min_value_bonus(bonuses, values):
    min_indices = locate_min([v for v in values])
    if not min_indices:
        return np.nan
    bonuses_lowest_values = np.array(bonuses)[min_indices]
    min_bonuses_index = np.nanargmin([abs(bonus) for bonus in bonuses_lowest_values])
    bonus_lowest_value = bonuses_lowest_values[min_bonuses_index]
    return bonus_lowest_value

def find_min_value_bonuses(intersectional_bonuses, values):
    min_indices = locate_min([v for v in values])
    if not min_indices:
        return np.nan
    bonuses_lowest_values = np.array(intersectional_bonuses)[min_indices]
    min_bonuses_index = np.nanargmin([sum([abs(bonus) for bonus in bonuses]) for bonuses in bonuses_lowest_values])
    bonus_lowest_value = bonuses_lowest_values[min_bonuses_index]
    bonus_lowest_value = tuple(bonus_lowest_value)
    return bonus_lowest_value

def locate_min(a):
    if np.isnan(a).all():
        return []
    smallest = np.nanmin(a)
    return [index for index, element in enumerate(a) if smallest == element]

def locate_max(a):
    if np.isnan(a).all():
        return []
    largest = np.nanmin(a)
    return [index for index, element in enumerate(a) if largest == element]