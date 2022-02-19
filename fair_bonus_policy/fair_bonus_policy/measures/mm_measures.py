#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:18:42 2019

@author: hertweck
"""

import numpy as np

def get_program_qualities(applications):
    sorted_programs = programs_quality(applications)
    quality_of_program = dict(sorted_programs)
    return sorted_programs, quality_of_program

def programs_quality(applications):
    pref_base = 'career_code_'
    wavg_base = 'weighted_average_score_'
    
    all_scores = []
    totals = {}
    
    for ii in range(10):
        pref_col = ''.join([pref_base, str(ii+1)])
        wavg_col = ''.join([wavg_base, str(ii+1)])
    
        preferences = applications[pref_col].values
        scores = applications[wavg_col].values
        for degree, scr in zip(preferences, scores):
            if not (np.isnan(degree) or np.isnan(scr)):
                all_scores.append(scr)
                
                tt, num, sos = totals.get(degree, (0, 0, 0))
                totals[degree] = (tt+scr, num+1, sos+scr**2)
    
    avges = [(degree, entry[0]/entry[1])  for degree, entry in totals.items()]
    sorted_schools = sorted(avges, key = lambda x: x[1], reverse=True)
    return sorted_schools

def admissions_utility(programs, students, quality_of_program):
    result = 0
    num_schools = 0
    for program_id, program in programs.items():
        admitted = program['accepted']
        if admitted:
            school_qual = quality_of_program[program_id]
            applicant_qual = np.mean([student['score_before_bonus'] for student in admitted])
            result += np.sqrt(applicant_qual * school_qual)
        num_schools += 1
    result /= num_schools
    return result

def admissions_utility_top_schools(universities, students, top_schools):
    result = 0
    num_schools = 0
    for institution_id in top_schools:
        admitted = universities[institution_id]['accepted']
        if admitted:
            applicant_qual = np.mean([student['score_before_bonus'] for student in admitted])
            result += applicant_qual
        num_schools += 1
    result /= num_schools
    return result

def admissions_utility_one_program(programs, program_id):
    admitted = programs[program_id]['accepted']
    if admitted:
        return round(np.mean([student['score_before_bonus'] for student in admitted]), 2)
    return 0


def admissions_disparity(universities, students, top_schools, normed : bool):
    total_adv = 2 # advantaged applicants
    total_dis = 2 # disadvantaged applicants
    top_adv = 1 # advantaged in top schools
    top_dis = 1 # disadvantaged in top schools
    for university in universities.values():
        admitted = university['accepted']
        num_dis = np.sum([students[student['student_id']]['ses'] == 'low' for student in admitted])
        num_adv = len(admitted) - num_dis
        total_adv += num_adv
        total_dis += num_dis
        if int(university['institution_id']) in top_schools:
            top_adv += num_adv
            top_dis += num_dis
    proportion_adv = top_adv / total_adv
    proportion_dis = top_dis / total_dis
    if normed is True:
        ZZ = (top_adv + top_dis) / (total_adv + total_dis)
        proportion_adv /= ZZ
        proportion_dis /= ZZ
    disparity = proportion_adv - proportion_dis
    return disparity

def admissions_disparity_eoo_ses(programs, students):
    total_disadvantaged = np.sum([student['ses'] == 'low' for student in students.values()])
    total_advantaged = len(students) - total_disadvantaged
    proportion_total_disadvantaged = total_disadvantaged / len(students)
    proportion_total_advantaged = total_advantaged / len(students)
    total_disparity = 0
    number_programs = 0
    for program in programs.values():
        admitted = program['accepted']
        if admitted:
            number_programs += 1
            program_disadvantaged = np.sum([students[student['student_id']]['ses'] == 'low' for student in admitted])
            program_advantaged = len(admitted) - program_disadvantaged
            proportion_program_disadvantaged = program_disadvantaged / len(admitted)
            proportion_program_advantaged = program_advantaged / len(admitted)
            representation_disadvantaged = proportion_program_disadvantaged - proportion_total_disadvantaged
            representation_advantaged = proportion_program_advantaged - proportion_total_advantaged
            program_disparity = abs(representation_advantaged - representation_disadvantaged)
            total_disparity += program_disparity
    total_disparity /= number_programs
    return total_disparity

def admissions_disparity_eoo(programs, students):
    total_disparity = 0
    number_programs = 0
    for program in programs.values():
        admitted = program['accepted']
        if admitted:
            number_programs += 1
            program_disadvantaged = np.sum([students[student['student_id']]['gender'] == 'f' for student in admitted])
            program_advantaged = len(admitted) - program_disadvantaged
            proportion_program_disadvantaged = program_disadvantaged / len(admitted)
            proportion_program_advantaged = program_advantaged / len(admitted)
            representation_disadvantaged = proportion_program_disadvantaged - 0.5
            representation_advantaged = proportion_program_advantaged - 0.5
            program_disparity = abs(representation_advantaged - representation_disadvantaged)
            total_disparity += program_disparity
    total_disparity /= number_programs
    return total_disparity

def admissions_disparity_eoo_one_program_ses(programs, program_id, students):
    total_disadvantaged = np.sum([student['ses'] == 'low' for student in students.values()])
    total_advantaged = len(students) - total_disadvantaged
    proportion_total_disadvantaged = total_disadvantaged / len(students)
    proportion_total_advantaged = total_advantaged / len(students)
    admitted = programs[program_id]['accepted']
    if admitted:
        program_disadvantaged = np.sum([students[student['student_id']]['ses'] == 'low' for student in admitted])
        program_advantaged = len(admitted) - program_disadvantaged
        proportion_program_disadvantaged = program_disadvantaged / len(admitted)
        proportion_program_advantaged = program_advantaged / len(admitted)
        representation_disadvantaged = proportion_program_disadvantaged - proportion_total_disadvantaged
        representation_advantaged = proportion_program_advantaged - proportion_total_advantaged
        return abs(representation_advantaged - representation_disadvantaged)
    return np.nan

def admissions_disparity_eoo_one_program(programs, program_id, students):
    admitted = programs[program_id]['accepted']
    if admitted:
        program_disadvantaged = np.sum([students[student['student_id']]['gender'] == 'f' for student in admitted])
        program_advantaged = len(admitted) - program_disadvantaged
        proportion_program_disadvantaged = program_disadvantaged / len(admitted)
        proportion_program_advantaged = program_advantaged / len(admitted)
        representation_disadvantaged = proportion_program_disadvantaged - 0.5
        representation_advantaged = proportion_program_advantaged - 0.5
        return abs(representation_advantaged - representation_disadvantaged)
    return np.nan

def count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged):
    admitted = programs[program_id]['accepted']
    if admitted:
        admitted_disadvantaged = np.sum([students[student['student_id']][column] == disadvantaged for student in admitted])
        admitted_advantaged = len(admitted) - admitted_disadvantaged
        return (admitted_disadvantaged, admitted_advantaged)
    return (0,0)

def count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged):
    applied = programs[program_id]['applied']
    if applied:
        applied_disadvantaged = np.sum([students[student['student_id']][column] == disadvantaged for student in applied])
        applied_advantaged = len(applied) - applied_disadvantaged
        return (applied_disadvantaged, applied_advantaged)
    return (0,0)

def count_intersectional_subgroups_for_admitted_students(programs, program_id, students, columns, disadvantaged_groups):
    admitted = programs[program_id]['accepted']
    if admitted:
        admitted_disadvantaged = 0
        for student in admitted:
            if all(students[student['student_id']][column] == disadvantaged for column, disadvantaged in zip(columns, disadvantaged_groups)):
                admitted_disadvantaged += 1
        admitted_advantaged = len(admitted) - admitted_disadvantaged
        return (admitted_disadvantaged, admitted_advantaged)
    return (0,0)

def count_intersectional_subgroups_for_applicants(programs, program_id, students, columns, disadvantaged_groups):
    applied = programs[program_id]['applied']
    if applied:
        applied_disadvantaged = 0
        for student in applied:
            if all(students[student['student_id']][column] == disadvantaged for column, disadvantaged in zip(columns, disadvantaged_groups)):
                applied_disadvantaged += 1
        applied_advantaged = len(applied) - applied_disadvantaged
        return (applied_disadvantaged, applied_advantaged)
    return (0,0)

def calculate_acceptance_rates(admitted_disadvantaged, admitted_advantaged, applied_disadvantaged, applied_advantaged):
    acceptance_rate_disadvantaged = admitted_disadvantaged / applied_disadvantaged if applied_disadvantaged != 0 else np.nan
    acceptance_rate_advantaged = admitted_advantaged / applied_advantaged if applied_advantaged != 0 else np.nan
    return (acceptance_rate_disadvantaged, acceptance_rate_advantaged)

def calculate_spd(disadvantaged, advantaged):
    spd = disadvantaged - advantaged
    return spd

def calculate_di(disadvantaged, advantaged):
    if disadvantaged == 0 and advantaged == 0:
        return 1
    elif advantaged == 0:
        return float('inf')
    return disadvantaged / advantaged

def statistical_parity_one_program(programs, program_id, students, column, disadvantaged):
    admitted_counts = count_subgroups_for_admitted_students(programs, program_id, students, column, disadvantaged)
    applied_counts = count_subgroups_for_applicants(programs, program_id, students, column, disadvantaged)
    acceptance_rates = calculate_acceptance_rates(*admitted_counts, *applied_counts)
    spd = calculate_spd(*acceptance_rates)
    di = calculate_di(*acceptance_rates)
    return (spd, di)

def statistical_parities_one_program(programs, program_id, students, columns, disadvantaged_groups):
    disparities = [statistical_parity_one_program(programs, program_id, students, column, disadvantaged) for column, disadvantaged in zip(columns, disadvantaged_groups)]
    return disparities

#def statistical_parity_one_program(programs, program_id, students, column, disadvantaged):
#    admitted = programs[program_id]['accepted']
#    applied = programs[program_id]['applied']
#    if admitted:
#        admitted_disadvantaged = np.sum([students[student['student_id']][column] == disadvantaged for student in admitted])
#        admitted_advantaged = len(admitted) - admitted_disadvantaged
#        applied_disadvantaged = np.sum([students[student['student_id']][column] == disadvantaged for student in applied])
#        applied_advantaged = len(applied) - applied_disadvantaged
#        acceptance_rate_disadvantaged = admitted_disadvantaged / applied_disadvantaged if applied_disadvantaged != 0 else 0
#        acceptance_rate_advantaged = admitted_advantaged / applied_advantaged if applied_advantaged != 0 else 0
#        spd = acceptance_rate_disadvantaged - acceptance_rate_advantaged
#        di = acceptance_rate_disadvantaged / acceptance_rate_advantaged if acceptance_rate_advantaged != 0 else float('inf')
#        return (spd, di)
#    return (0, 1)

def get_all_applicants_column_for(program_id, students, column):
    genders = []
    for student in students.values():
        for preference in student['preferences'].values():
            if preference['institution_id'] == program_id:
                genders.append(student[column])
                break
    return genders

def lowest_scores(programs, with_bonus=False):
    lowest_scores = {}
    for institution_id, program in programs.items():
        admitted = program['accepted']
        if admitted:
            if with_bonus:
                lowest = np.amin([student['score'] for student in admitted])
            else:
                lowest = np.amin([student['score_before_bonus'] for student in admitted])
            lowest_scores[institution_id] = lowest
    return lowest_scores  

def lowest_score(program_id, programs, with_bonus=True):
    lowest_score = 0
    admitted = programs[program_id]['accepted']
    if admitted:
        if with_bonus:
            lowest = np.amin([student['score'] for student in admitted])
        else:
            lowest = np.amin([student['score_before_bonus'] for student in admitted])
        lowest_score = lowest
    return lowest_score

def objective_function(bonus_values, utility, spd, _lambda):
    zero_index, = np.where(bonus_values == 0)
    zero_index = zero_index[0]
    original_utility = utility[zero_index]
    y = []
    utility_difference_y = []
    disparity_y = []
    for u, p in zip(utility, spd):
        results = objective_function_point(u, original_utility, [p], [_lambda])
        y.append(results[0])
        utility_difference_y.append(results[1])
        disparity_y.append(results[2])
#    utility_difference = utility[zero_index] - np.array(utility)
#    utility_difference_y = np.array(utility_difference)
#    disparity_y = abs(np.array(statistical_parity)) * _lambda
#    y = utility_difference_y + disparity_y
    return y, utility_difference_y, disparity_y

def objective_function_multiple_attributes(bonus_values, utility, spds, _lambdas):
    zero_index, = np.where(bonus_values == 0)
    zero_index = zero_index[0]
    original_utility = utility[zero_index]
    y = []
    utility_difference_y = []
    disparity_y = []
    for i in range(len(utility)):
        utility_point = utility[i]
        spds_point = [spd[i] for spd in spds]
        results = objective_function_point(utility_point, original_utility, spds_point, _lambdas)
        y.append(results[0])
        utility_difference_y.append(results[1])
        disparity_y.append(results[2])
    return y, utility_difference_y, disparity_y

def objective_function_point(utility, original_utility, spds, _lambdas):
    utility_difference = original_utility - utility
    disparity = sum([abs(spd) * _lambda for spd, _lambda in zip(spds, _lambdas)])
    objective = utility_difference + disparity
    return objective, utility_difference, disparity