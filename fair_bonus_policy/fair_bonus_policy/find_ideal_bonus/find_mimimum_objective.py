#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb  1 23:38:27 2020

@author: hertweck
"""

import numpy as np
from fair_bonus_policy.measures import evaluate_policy as ep
from fair_bonus_policy.utils import minimum
import copy

def find_minimum_objective(program_id, _lambda, original_students, original_programs, column, disadvantaged, disparity_measure='spd', utility_measure='utility'):
    students = copy.deepcopy(original_students)
    programs = copy.deepcopy(original_programs)
    # The minimum of the objective function has to be between the minimum of the utility_difference (0) and the minimum disparity
    range_max, bonus_values = find_single_minimum(program_id, _lambda, students, programs, column, disadvantaged, 'disparity', disparity_measure, utility_measure)
    
    range_min = 0
    if range_max > 0:
        range_min = -2
    elif range_max < 0:
        range_min = 2
        
    for bonus in range(min(range_min,range_max), max(range_min,range_max)):
        bonus_values = ep.evaluate_bonus_policy_same_year(bonus, program_id, _lambda, students, programs, column, disadvantaged, disparity_measure, utility_measure, bonus_values)
    added_bonuses, added_objectives = create_bonus_metric_lists(bonus_values, 'objective')
    ideal_bonus = minimum.find_min_value_bonus(added_bonuses, added_objectives)
    return ideal_bonus, bonus_values
    

def find_single_minimum(program_id, _lambda, original_students, original_programs, column, disadvantaged, metric_to_minimize='disparity', disparity_measure='spd', utility_measure='utility'):
    students = copy.deepcopy(original_students)
    programs = copy.deepcopy(original_programs)
    
    bonus_values = ep.evaluate_bonus_policy_same_year(0, program_id, _lambda, students, programs, column, disadvantaged, disparity_measure, utility_measure, {})
    ideal_bonus = np.nan
    
    stepsize = 64
    range_min = -120
    range_max = 120
    
    while np.isnan(ideal_bonus):
        last_same_results_as_range_min = range_min
        last_same_results_as_range_max = range_max
        bonus_values = ep.evaluate_bonus_policy_same_year(range_min, program_id, _lambda, students, programs, column, disadvantaged, disparity_measure, utility_measure, bonus_values)
        bonus_values = ep.evaluate_bonus_policy_same_year(range_max, program_id, _lambda, students, programs, column, disadvantaged, disparity_measure, utility_measure, bonus_values)
        previous_metric = bonus_values[range_min][metric_to_minimize]
        bonus = range_min
        while bonus < range_max and bonus_values[bonus][metric_to_minimize] <= previous_metric:
            previous_metric = bonus_values[bonus][metric_to_minimize]
            bonus += stepsize
            bonus_values = ep.evaluate_bonus_policy_same_year(bonus, program_id, _lambda, students, programs, column, disadvantaged, disparity_measure, utility_measure, bonus_values)
            if bonus_values[last_same_results_as_range_min]['assignments'] == bonus_values[bonus]['assignments']:
                last_same_results_as_range_min = bonus
            if last_same_results_as_range_max == range_max and bonus_values[last_same_results_as_range_max]['assignments'] == bonus_values[bonus]['assignments']:
                last_same_results_as_range_max = bonus
        range_min = last_same_results_as_range_min
        range_max = min(bonus, last_same_results_as_range_max)
        added_bonuses, added_metrics = create_bonus_metric_lists(bonus_values, metric_to_minimize)
        if range_min >= range_max or stepsize == 1:
            # This means the original assignment never changed, so we have a flat line
            # Or the step size is 1, so we have already checked all points
            # We simply pick the lowest absolute bonus
            ideal_bonus = minimum.find_min_value_bonus(added_bonuses, added_metrics)
            break
        if len(added_metrics) >= 4:
            min_objective_indices = minimum.locate_min(added_metrics)
            range_min_index = min_objective_indices[0] - 1 if (min_objective_indices[0] - 1) >= 0 else 0
            range_min = max(range_min, added_bonuses[range_min_index])
            last_index = len(added_metrics) - 1
            range_max_index = min_objective_indices[-1] + 1 if (min_objective_indices[-1] + 1) <= last_index else last_index
            range_max = min(range_max, added_bonuses[range_max_index])
        if range_max - range_min <= 1:
            ideal_bonus = minimum.find_min_value_bonus(added_bonuses, added_metrics)
            break
        stepsize = decrease_stepsize(stepsize)
    return ideal_bonus, bonus_values

def create_bonus_metric_lists(bonus_values, metric):
    bonus_metric_pairs = [(key, metrics[metric]) for key, metrics in bonus_values.items()]
    bonus_metric_pairs = sorted(bonus_metric_pairs, key=lambda tup: tup[0])
    added_bonuses = [tup[0] for tup in bonus_metric_pairs]
    added_metrics = [tup[1] for tup in bonus_metric_pairs]
    return added_bonuses, added_metrics

def decrease_stepsize(stepsize):
    if stepsize > 1:
        stepsize = stepsize // 2
    return stepsize