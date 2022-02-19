#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar  9 15:47:44 2020

@author: hertweck
"""

from fair_bonus_policy.measures import evaluate_policy as ep

def calculate_baseline(programs_sorted_by_quality, ideal, baseline, students_ideal, programs_ideal, _lambda=5, column='gender', disadvantaged='f', disparity_measure='spd', utility_measure='utility'):
    evaluations = {}
    for program_id in programs_sorted_by_quality:
        print(program_id)
        if program_id in baseline and program_id in ideal and isinstance(ideal[program_id], dict):
            baseline_bonus = baseline[program_id]
            original_utility = ideal[program_id]['zero_metrics']['utility']
            predicted_metrics = ep.evaluate_bonus_policy(baseline_bonus, program_id, original_utility, _lambda, students_ideal, programs_ideal, column, disadvantaged, disparity_measure, utility_measure)
            evaluations[program_id] = {'ideal_bonus': baseline_bonus, 'ideal_metrics': predicted_metrics}
    return evaluations

def compare_strategy_to_ideal(ideal, strategy, programs_ideal):
    evaluations = {}
    for program_id in programs_ideal:
        if program_id in strategy and isinstance(ideal[program_id],dict):
            evaluation = compare_points_for_single_program(program_id, ideal, strategy)
            evaluations[program_id] = evaluation
    return evaluations


def compare_points_for_single_program(program_id, ideal, strategy):
    evaluation = {}
    ideal_bonus = ideal[program_id]['ideal_bonus']
    ideal_metrics = ideal[program_id]['ideal_metrics']
    zero_metrics = ideal[program_id]['zero_metrics']
    strategy_bonus = strategy[program_id]['ideal_bonus']
    strategy_metrics = strategy[program_id]['ideal_metrics']
    bonus_difference = strategy_bonus - ideal_bonus
    objective_difference = strategy_metrics['objective'] - ideal_metrics['objective']
    utility_loss_difference = strategy_metrics['utility_difference'] - ideal_metrics['utility_difference']
    disparity_difference = abs(strategy_metrics['spd']) - abs(ideal_metrics['spd'])
    if zero_metrics['spd'] < 0:
        disparity_difference_direction = ideal_metrics['spd'] - strategy_metrics['spd']
    elif zero_metrics['spd'] > 0:
        disparity_difference_direction = strategy_metrics['spd'] - ideal_metrics['spd']
    else:
        disparity_difference_direction = abs(strategy_metrics['spd'])
    if zero_metrics['admitted_disadvantaged'] < zero_metrics['admitted_advantaged']:
        disparity_difference_underrepresentation = ideal_metrics['spd'] - strategy_metrics['spd']
    elif zero_metrics['admitted_disadvantaged'] > zero_metrics['admitted_advantaged']:
        disparity_difference_underrepresentation = strategy_metrics['spd'] - ideal_metrics['spd']
    else:
        disparity_difference_underrepresentation = abs(strategy_metrics['spd'])
    
    objective_difference_control = strategy_metrics['objective'] - zero_metrics['objective']
    utility_loss_difference_control = strategy_metrics['utility_difference'] - zero_metrics['utility_difference']
    absolute_disparity_difference_control = abs(strategy_metrics['spd']) - abs(zero_metrics['spd'])
    if zero_metrics['spd'] < 0:
        disparity_difference_control = zero_metrics['spd'] - strategy_metrics['spd']
    elif zero_metrics['spd'] > 0:
        disparity_difference_control = strategy_metrics['spd'] - zero_metrics['spd']
    else:
        disparity_difference_control = abs(strategy_metrics['spd'])
    if zero_metrics['admitted_disadvantaged'] < zero_metrics['admitted_advantaged']:
        disparity_difference_control_underrepresentation = zero_metrics['spd'] - strategy_metrics['spd']
    elif zero_metrics['admitted_disadvantaged'] > zero_metrics['admitted_advantaged']:
        disparity_difference_control_underrepresentation = strategy_metrics['spd'] - zero_metrics['spd']
    else:
        disparity_difference_control_underrepresentation = abs(strategy_metrics['spd'])

    evaluation['bonus_difference'] = bonus_difference
    evaluation['ideal_disparitiy'] = ideal_metrics['spd']
    evaluation['predicted_disparity'] = strategy_metrics['spd']
    evaluation['objective_difference'] = objective_difference
    evaluation['utility_loss_difference'] = utility_loss_difference
    evaluation['disparity_difference'] = disparity_difference
    evaluation['disparity_difference_direction'] = disparity_difference_direction
    evaluation['disparity_difference_underrepresentation'] = disparity_difference_underrepresentation
    evaluation['objective_difference_control'] = objective_difference_control
    evaluation['utility_loss_difference_control'] = utility_loss_difference_control
    evaluation['absolute_disparity_difference_control'] = absolute_disparity_difference_control
    evaluation['disparity_difference_control'] = disparity_difference_control
    evaluation['disparity_difference_control_underrepresentation'] = disparity_difference_control_underrepresentation
    
    return evaluation

def compare_intersectional_strategy_to_ideal(ideal, independent, strategy, programs_ideal):
    evaluations = {}
    for program_id in programs_ideal:
        if program_id in strategy and isinstance(strategy[program_id], dict)\
        and program_id in ideal and isinstance(ideal[program_id], dict)\
        and program_id in independent and isinstance(independent[program_id], dict):
            evaluation = compare_intersectional_points_for_single_program(program_id, ideal, independent, strategy)
            evaluations[program_id] = evaluation
    return evaluations

def compare_intersectional_points_for_single_program(program_id, ideal, independent, strategy):
    evaluation = {}
    ideal_bonuses = ideal[program_id]['ideal_bonuses']
    ideal_metrics = ideal[program_id]['ideal_metrics']
    zero_metrics = independent[program_id]['zero_metrics']
    strategy_bonuses = strategy[program_id]['ideal_bonuses']
    strategy_metrics = strategy[program_id]['ideal_metrics']
    
    bonus_difference = sum([abs(b) for b in strategy_bonuses]) - sum([abs(b) for b in ideal_bonuses])
    gender_bonus_difference = strategy_bonuses[0] - ideal_bonuses[0]
    ses_bonus_difference = strategy_bonuses[1] - ideal_bonuses[1]
    objective_difference = strategy_metrics['objective'] - ideal_metrics['objective']
    utility_loss_difference = strategy_metrics['utility_difference'] - ideal_metrics['utility_difference']
    disparity_difference = strategy_metrics['disparity'] - ideal_metrics['disparity']
    objective_difference_control = strategy_metrics['objective'] - zero_metrics['objective']
    utility_loss_difference_control = strategy_metrics['utility_difference'] - zero_metrics['utility_difference']
    disparity_difference_control = strategy_metrics['disparity'] - zero_metrics['disparity']
    
    absolute_spd_gender_difference_control = abs(strategy_metrics['spd_gender']) - abs(zero_metrics['spd_gender'])
    if zero_metrics['spd_gender'] < 0:
        spd_gender_difference_control = zero_metrics['spd_gender'] - strategy_metrics['spd_gender']
    elif zero_metrics['spd_gender'] > 0:
        spd_gender_difference_control = strategy_metrics['spd_gender'] - zero_metrics['spd_gender']
    else:
        spd_gender_difference_control = abs(strategy_metrics['spd_gender'])
    
    if zero_metrics['admitted_disadvantaged_gender'] < zero_metrics['admitted_advantaged_gender']:
        spd_gender_difference_control_underrepresenation = zero_metrics['spd_gender'] - strategy_metrics['spd_gender']
    elif zero_metrics['admitted_disadvantaged_gender'] > zero_metrics['admitted_advantaged_gender']:
        spd_gender_difference_control_underrepresenation = strategy_metrics['spd_gender'] - zero_metrics['spd_gender']
    else:
        spd_gender_difference_control_underrepresenation = abs(strategy_metrics['spd_gender'])
        
    absolute_spd_ses_difference_control = abs(strategy_metrics['spd_ses']) - abs(zero_metrics['spd_ses'])
    if zero_metrics['spd_ses'] < 0:
        spd_ses_difference_control = zero_metrics['spd_ses'] - strategy_metrics['spd_ses']
    elif zero_metrics['spd_ses'] > 0:
        spd_ses_difference_control = strategy_metrics['spd_ses'] - zero_metrics['spd_ses']
    else:
        spd_ses_difference_control = abs(strategy_metrics['spd_ses'])
        
    if zero_metrics['admitted_disadvantaged_ses'] < zero_metrics['admitted_advantaged_ses']:
        spd_ses_difference_control_underrepresentation = zero_metrics['spd_ses'] - strategy_metrics['spd_ses']
    elif zero_metrics['admitted_disadvantaged_ses'] > zero_metrics['admitted_advantaged_ses']:
        spd_ses_difference_control_underrepresentation = strategy_metrics['spd_ses'] - zero_metrics['spd_ses']
    else:
        spd_ses_difference_control_underrepresentation = abs(strategy_metrics['spd_ses'])

    evaluation['bonus_difference'] = bonus_difference
    evaluation['gender_bonus_difference'] = gender_bonus_difference
    evaluation['ses_bonus_difference'] = ses_bonus_difference
    evaluation['objective_difference'] = objective_difference
    evaluation['utility_loss_difference'] = utility_loss_difference
    evaluation['disparity_difference'] = disparity_difference
    evaluation['objective_difference_control'] = objective_difference_control
    evaluation['utility_loss_difference_control'] = utility_loss_difference_control
    evaluation['disparity_difference_control'] = disparity_difference_control
    evaluation['absolute_spd_gender_difference_control'] = absolute_spd_gender_difference_control
    evaluation['spd_gender_difference_control'] = spd_gender_difference_control
    evaluation['spd_gender_difference_control_underrepresenation'] = spd_gender_difference_control_underrepresenation
    evaluation['absolute_spd_ses_difference_control'] = absolute_spd_ses_difference_control
    evaluation['spd_ses_difference_control'] = spd_ses_difference_control
    evaluation['spd_ses_difference_control_underrepresentation'] = spd_ses_difference_control_underrepresentation

    return evaluation