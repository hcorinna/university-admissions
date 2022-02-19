#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 16:27:19 2019

@author: hertweck
"""

from fair_bonus_policy.clean_data import prepare_data
from fair_bonus_policy.measures import mm_measures
from fair_bonus_policy.plotting import bonus_values_vs_measures_plots as bvmp

# =============================================================================
# Plot utility vs disparity for a single program
# =============================================================================

def calculate_statistical_parity_one_program(program_id, run_matching_algorithm, students, programs, bonus_values, column, disadvantaged):
    utility = []
    spd = []
    di = []
    cutoff = []
    for bonus in bonus_values:
        prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
        students, programs = run_matching_algorithm(students, programs)
        utility_program = mm_measures.admissions_utility_one_program(programs, program_id)
        spd_program, di_program = mm_measures.statistical_parity_one_program(programs, program_id, students, column, disadvantaged)
        print('SPD:', spd_program, 'DI:', di_program)
        cutoff_program = mm_measures.lowest_score(program_id, programs, with_bonus=False)
        utility.append(utility_program)
        spd.append(spd_program)
        di.append(di_program)
        cutoff.append(cutoff_program)
    return utility, spd, di, cutoff

def calculate_statistical_parity_for_programs(run_matching_algorithm, students, programs, bonus_values, column, disadvantaged, program_keys=None):
    program_measures = {}
    if program_keys is None:
        program_keys = programs.keys()
    for program_id in program_keys:
        if program_id in programs.keys():
            # this is False when the given program_keys are e.g. from 2016, but we're evaluating on 2015 data
            program_measures[program_id] = {
                'utility': [],
                'spd': [],
                'di': [],
    #            'cutoff': [],
                'bonus_values': bonus_values
            }
        for bonus in bonus_values:
            print('bonus:', bonus)
            prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
            students, programs = run_matching_algorithm(students, programs)
            utility_program = mm_measures.admissions_utility_one_program(programs, program_id)
            spd_program, di_program = mm_measures.statistical_parity_one_program(programs, program_id, students, column, disadvantaged)
#            cutoff_program = mm_measures.lowest_score(program_id, programs, with_bonus=False)
            program_measures[program_id]['utility'].append(utility_program)
            program_measures[program_id]['spd'].append(spd_program)
            program_measures[program_id]['di'].append(di_program)
#            program_measures[program_id]['cutoff'].append(cutoff_program)
    return program_measures
    
# =============================================================================
# Plot utility vs disparity for all programs
# =============================================================================

def utility_disparity_all_programs(title, run_matching_algorithm, students, programs, bonus_values, quality_of_school):
    utility, disparity = calculate_disparity_utility_all_programs(run_matching_algorithm, students, programs, bonus_values, quality_of_school)
    bvmp.plot_utility(title, bonus_values, utility)
    bvmp.plot_disparity(title, bonus_values, disparity)

def calculate_disparity_utility_all_programs(run_matching_algorithm, students, programs, bonus_values, quality_of_school):
    utility = []
    disparity = []
    for bonus in bonus_values:
        prepare_data.apply_bonus(students, bonus)
        students, universities = run_matching_algorithm(students, programs)
        utility_bonus = mm_measures.admissions_utility(programs, students, quality_of_school)
        disparity_bonus = mm_measures.admissions_disparity_eoo(programs, students)
        utility.append(utility_bonus)
        disparity.append(disparity_bonus)
    return utility, disparity

# =============================================================================
# Results of matching in numbers
# =============================================================================

def results_of_students(students):
    students_without_application = 0
    students_without_place = 0
    students_with_place = 0
    students_with_first_choice = 0
    students_with_second_choice = 0
    students_with_third_choice = 0
    for student in students.values():
        if len(student['preferences']) == 0:
            students_without_application += 1
        if student['status'] == 'none':
            students_without_place += 1
        else:
            students_with_place += 1
            if student['next_preferred_school'] == 1:
                students_with_first_choice += 1
            elif student['next_preferred_school'] == 2:
                students_with_second_choice += 1
            elif student['next_preferred_school'] == 3:
                students_with_third_choice += 1
    
    print("Students that have not sent an application:", students_without_application)
    print("Students that have not been accepted:", students_without_place)
    print("Students that have been accepted:", students_with_place)
    print("Students that have been accepted at their 1st choice:", students_with_first_choice)
    print("Students that have been accepted at their 2nd choice:", students_with_second_choice)
    print("Students that have been accepted at their 3rd choice:", students_with_third_choice)
    
def results_of_programs(programs):
    spaces_left = 0
    quota_filled = 0
    overflow = 0
    for program in programs.values():
        if len(program['accepted']) < program['quota']:
            spaces_left += 1
        elif len(program['accepted']) == program['quota']:
            quota_filled += 1
        else:
            overflow += 1
    print("Programs that have spaces left:", spaces_left)
    print("Programs that have filled their quota:", quota_filled)
    print("Programs that have accepted too many students (should not happen):", overflow)