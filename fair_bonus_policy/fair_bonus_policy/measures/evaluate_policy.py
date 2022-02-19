#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 19:19:22 2020

@author: hertweck
"""

from fair_bonus_policy.clean_data import prepare_data
from fair_bonus_policy.measures import mm_measures
from fair_bonus_policy.matching_algorithms import deferred_acceptance as da


def evaluate_bonus_policy(bonus, program_id, original_utility, _lambda, students, programs, column, disadvantaged,
                          disparity_measure='spd', utility_measure='utility'):
    # When the policy is to be evaluated on another application set than it was optimized on, e.g. optimized for 2016, now evaluated on 2017
    prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    utility = mm_measures.admissions_utility_one_program(programs, program_id)
    cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    admitted_counts = mm_measures.count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
    applied_counts = mm_measures.count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
    acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
    spd = mm_measures.calculate_spd(*acceptance_rates)
    di = mm_measures.calculate_di(*acceptance_rates)
    metrics = {
            'admitted_disadvantaged': admitted_counts[0],
            'admitted_advantaged': admitted_counts[1],
            'utility': utility,
            'cutoff': cutoff,
            'spd': spd,
            'di': di
            }
    objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                    original_utility,
                                                                                    [metrics[disparity_measure]],
                                                                                    [_lambda])
    metrics['objective'] = objective
    metrics['utility_difference'] = utility_difference
    metrics['disparity'] = disparity
    prepare_data.remove_bonuses(students)
    return metrics

def evaluate_bonus_policy_same_year(bonus, program_id, _lambda, students, programs, column, disadvantaged,
                          disparity_measure='spd', utility_measure='utility', bonus_values={}):
    # When the policy is to be optimized and the original_utility thus comes from the same year
    if bonus not in bonus_values:
        prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
        students, programs = da.execute(students, programs)
        utility = mm_measures.admissions_utility_one_program(programs, program_id)
        cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
        admitted_counts = mm_measures.count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
        applied_counts = mm_measures.count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
        acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
        spd = mm_measures.calculate_spd(*acceptance_rates)
        di = mm_measures.calculate_di(*acceptance_rates)
        metrics = {
                'admitted_disadvantaged': admitted_counts[0],
                'admitted_advantaged': admitted_counts[1],
                'utility': utility,
                'cutoff': cutoff,
                'spd': spd,
                'di': di,
                'assignments': get_assignments(programs)
                }
        original_utility = metrics[utility_measure] if bonus == 0 else bonus_values[0][utility_measure]
        objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                        original_utility,
                                                                                        [metrics[disparity_measure]],
                                                                                        [_lambda])
        metrics['objective'] = objective
        metrics['utility_difference'] = utility_difference
        metrics['disparity'] = disparity
        bonus_values[bonus] = metrics
        prepare_data.remove_bonuses(students)
    return bonus_values

def evaluate_single_policy_on_intersectional_group(bonus, program_id, _lambda,
                                                   students, programs, column, disadvantaged,
                                                   disparity_measure='spd', utility_measure='utility', bonus_values={}):
    prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    utility = mm_measures.admissions_utility_one_program(programs, program_id)
    cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    metrics = {
        'utility': utility,
        'cutoff': cutoff
    }
    original_utility = metrics[utility_measure] if bonus == 0 else bonus_values[0][utility_measure]
    metrics = evaluate_policy_on_intersectional_groups(metrics, original_utility, program_id, _lambda, students, programs,
                                                   disparity_measure, utility_measure)
    bonus_values[bonus] = metrics
    prepare_data.remove_bonuses(students)
    return bonus_values

def evaluate_intersectional_policy_on_intersectional_group(bonuses, program_id, _lambda,
                                                   students, programs, columns, disadvantaged_groups,
                                                   disparity_measure='spd', utility_measure='utility', bonus_values={}):
    for bonus, column, disadvantaged in zip(bonuses, columns, disadvantaged_groups):
            prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    utility = mm_measures.admissions_utility_one_program(programs, program_id)
    cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    metrics = {
        'utility': utility,
        'cutoff': cutoff
    }
    original_utility = metrics[utility_measure] if bonuses == (0,0) else bonus_values[(0,0)][utility_measure]
    metrics = evaluate_policy_on_intersectional_groups(metrics, original_utility, program_id, _lambda, students, programs,
                                                   disparity_measure, utility_measure)
    bonus_values[bonuses] = metrics
    prepare_data.remove_bonuses(students)
    return bonus_values

def evaluate_policy_on_intersectional_groups(metrics, original_utility, program_id, _lambda, students, programs,
                                                   disparity_measure='spd', utility_measure='utility'):
    columns = ['gender', 'ses']
    disadvantaged_group_combinations = [[g, s] for g in ['f', 'm'] for s in ['low', 'high']]
    for disadvantaged_groups in disadvantaged_group_combinations:
        admitted_counts = mm_measures.count_intersectional_subgroups_for_admitted_students(programs, program_id,
                                                                                           students, columns,
                                                                                           disadvantaged_groups)
        applied_counts = mm_measures.count_intersectional_subgroups_for_applicants(programs, program_id, students,
                                                                                   columns, disadvantaged_groups)
        acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
        spd = mm_measures.calculate_spd(*acceptance_rates)
        di = mm_measures.calculate_di(*acceptance_rates)
        metrics_intersectional_group = {
                'admitted_disadvantaged': admitted_counts[0],
                'admitted_advantaged': admitted_counts[1],
                'spd': spd,
                'di': di
                }
        objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                        original_utility,
                                                                                        [metrics_intersectional_group[disparity_measure]],
                                                                                        [_lambda])
        metrics_intersectional_group['objective'] = objective
        metrics_intersectional_group['utility_difference'] = utility_difference
        metrics_intersectional_group['disparity'] = disparity
        metrics[tuple(disadvantaged_groups)] = metrics_intersectional_group
    return metrics

def evaluate_intersectional_bonus_policy(bonuses, program_id, _lambdas, students, programs, columns,
                                         disadvantaged_groups, disparity_measure='spd', utility_measure='utility',
                                         bonus_values={}):
    if bonuses not in bonus_values:
        for bonus, column, disadvantaged in zip(bonuses, columns, disadvantaged_groups):
            prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
        students, programs = da.execute(students, programs)
        utility = mm_measures.admissions_utility_one_program(programs, program_id)
        cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
        metrics = {
                'utility': utility,
                'cutoff': cutoff
                }
        for column, disadvantaged in zip(columns, disadvantaged_groups):
            admitted_counts = mm_measures.count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
            applied_counts = mm_measures.count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
            acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
            spd = mm_measures.calculate_spd(*acceptance_rates)
            di = mm_measures.calculate_di(*acceptance_rates)
            metrics['admitted_disadvantaged_' + column] = admitted_counts[0]
            metrics['admitted_advantaged_' + column] = admitted_counts[1]
            metrics['spd_' + column] = spd
            metrics['di_' + column] = di
        original_utility = metrics[utility_measure] if bonuses == (0,0) else bonus_values[(0,0)][utility_measure]
        # Put all statistical parity measures in one list
        statistical_disparities = [metrics[disparity_measure + '_' + column] for column in columns]
        objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                        original_utility,
                                                                                        statistical_disparities,
                                                                                        _lambdas)
        metrics['objective'] = objective
        metrics['utility_difference'] = utility_difference
        metrics['disparity'] = disparity
        bonus_values[bonuses] = metrics
        prepare_data.remove_bonuses(students)
    return bonus_values

def evaluate_zero_metrics_intersectional_bonus_policy_multiple_years(program_id, _lambdas, students, programs, columns,
                                         disadvantaged_groups, disparity_measure='spd', utility_measure='utility'):
    students, programs = da.execute(students, programs)
    utility = mm_measures.admissions_utility_one_program(programs, program_id)
    cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    metrics = {
            'utility': utility,
            'cutoff': cutoff
            }
    for column, disadvantaged in zip(columns, disadvantaged_groups):
        admitted_counts = mm_measures.count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
        applied_counts = mm_measures.count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
        acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
        spd = mm_measures.calculate_spd(*acceptance_rates)
        di = mm_measures.calculate_di(*acceptance_rates)
        metrics['admitted_disadvantaged_' + column] = admitted_counts[0]
        metrics['admitted_advantaged_' + column] = admitted_counts[1]
        metrics['spd_' + column] = spd
        metrics['di_' + column] = di
    # Put all statistical parity measures in one list
    statistical_disparities = [metrics[disparity_measure + '_' + column] for column in columns]
    objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                    metrics[utility_measure],
                                                                                    statistical_disparities,
                                                                                    _lambdas)
    metrics['objective'] = objective
    metrics['utility_difference'] = utility_difference
    metrics['disparity'] = disparity
    return metrics

def evaluate_intersectional_bonus_policy_multiple_years(bonuses, program_id, original_utility, _lambdas, students, programs, columns,
                                         disadvantaged_groups, disparity_measure='spd', utility_measure='utility'):
    for bonus, column, disadvantaged in zip(bonuses, columns, disadvantaged_groups):
        prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    utility = mm_measures.admissions_utility_one_program(programs, program_id)
    cutoff = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    metrics = {
            'utility': utility,
            'cutoff': cutoff
            }
    for column, disadvantaged in zip(columns, disadvantaged_groups):
        admitted_counts = mm_measures.count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
        applied_counts = mm_measures.count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
        acceptance_rates = mm_measures.calculate_acceptance_rates(*admitted_counts, *applied_counts)
        spd = mm_measures.calculate_spd(*acceptance_rates)
        di = mm_measures.calculate_di(*acceptance_rates)
        metrics['admitted_disadvantaged_' + column] = admitted_counts[0]
        metrics['admitted_advantaged_' + column] = admitted_counts[1]
        metrics['spd_' + column] = spd
        metrics['di_' + column] = di
    # Put all statistical parity measures in one list
    statistical_disparities = [metrics[disparity_measure + '_' + column] for column in columns]
    objective, utility_difference, disparity = mm_measures.objective_function_point(metrics[utility_measure],
                                                                                    original_utility,
                                                                                    statistical_disparities,
                                                                                    _lambdas)
    metrics['objective'] = objective
    metrics['utility_difference'] = utility_difference
    metrics['disparity'] = disparity
    prepare_data.remove_bonuses(students)
    return metrics

def get_assignments(programs):
    return {program_id:{student['student_id'] for student in program['accepted']} for program_id, program in programs.items()}