#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 27 12:39:04 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
import math
from fair_bonus_policy.clean_data import prepare_data
from fair_bonus_policy.measures import mm_measures
from fair_bonus_policy.matching_algorithms import deferred_acceptance as da
from fair_bonus_policy.clean_data.ClassificationPreparer import ClassificationPreparer
from fair_bonus_policy.application_predictions.Predictor import OneTopicPerApplicationLogisticRegressionPredictor
import copy
import cProfile


def find_ideal_bonus_gender(program_id, cutoff_loss, students, programs):
    bonus_values = {}
    students = copy.deepcopy(students)
    programs = copy.deepcopy(programs)
    
    students, programs = da.execute(students, programs)
    disparity_no_bonus = mm_measures.admissions_disparity_eoo_one_program(programs, program_id, students)
    cutoff_no_bonus = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    bonus_values[0] = {'disparity': disparity_no_bonus, 'cutoff': cutoff_no_bonus}
    
    if bonus_values[0]['disparity'] < 0.1 or cutoff_loss == 0:
        return 0, None
    
    disadvantaged_gender, label = which_gender_is_disadvantaged(program_id, students, programs)
    min_cutoff_constraint = bonus_values[0]['cutoff'] - cutoff_loss
    ideal_bonus = np.nan
    
    stepsize = 50
    range_min = 0
    range_max = 100
    
    
    while np.isnan(ideal_bonus):
        bonus = range_min
        added_bonuses = [bonus]
        added_disparities = [bonus_values[bonus]['disparity']]
        previous_disparity = bonus_values[bonus]['disparity']
        last_same_results = (bonus, get_assignments(programs))
        while bonus < range_max and bonus_values[bonus]['cutoff'] > min_cutoff_constraint and bonus_values[bonus]['disparity'] <= previous_disparity:
            previous_disparity = bonus_values[bonus]['disparity']
            bonus += stepsize
            add_calculations_to_bonus_values(bonus_values, bonus, disadvantaged_gender, program_id, students, programs)
            added_bonuses.append(bonus)
            added_disparities.append(bonus_values[bonus]['disparity'])
            if not bonus_values[bonus] == bonus_values[last_same_results[0]] or not last_same_results[1] == get_assignments(programs):
                last_same_results = (bonus, get_assignments(programs))
        min_disparity_index = np.argmin(added_disparities)
        if stepsize == 1:
            if bonus_values[bonus]['cutoff'] < min_cutoff_constraint:
                min_disparity_index = np.argmin(added_disparities[:len(added_disparities)-1])
            ideal_bonus = added_bonuses[min_disparity_index]
            break
        range_max = last_same_results[0]
        # Is the last tested bonus the one that produces the smallest disparity?
        if min_disparity_index == (len(added_disparities) - 1):
            # Pick minimum of last tested bonus and range_max
            range_max = min(bonus, range_max)
            # If we have at least four points and the first one has a higher value than the second to last one, then we can simply search between the second to last and last one
            if len(added_disparities) >= 4 and added_disparities[0] > added_disparities[min_disparity_index - 1]:
                range_min = added_bonuses[min_disparity_index - 1]
            if range_max - range_min <= 1:
                # TODO: or range_max? Shouldn't we compare their disparities?
                ideal_bonus = range_min
                break
        elif min_disparity_index > 0:
            range_min = added_bonuses[min_disparity_index - 1]
            range_max_disparity = added_disparities[min_disparity_index + 1]
            same_disparities = 0
            while range_max_disparity == added_disparities[min_disparity_index] and min_disparity_index + 1 + same_disparities < len(added_disparities) - 1:
                same_disparities += 1
                range_max_disparity = added_disparities[min_disparity_index + 1 + same_disparities]
            range_max = added_bonuses[min_disparity_index + 1 + same_disparities] # min previous range_max
            if range_max - range_min <= 2 + same_disparities:
                ideal_bonus = added_bonuses[min_disparity_index]
                break
        stepsize = decrease_stepsize(stepsize)
    return ideal_bonus, disadvantaged_gender
    
def decrease_stepsize(stepsize):
    if stepsize > 1:
        stepsize = stepsize // 2
    return stepsize

def get_assignments(programs):
    return {program_id:{student['student_id'] for student in program['accepted']} for program_id, program in programs.items()}

def calculate_disparity_cutoff(bonus, disadvantaged_gender, program_id, students, programs):
    prepare_data.apply_gender_bonus_program(students, bonus, disadvantaged_gender, program_id)
    students, programs = da.execute(students, programs)
    disparity_bonus = mm_measures.admissions_disparity_eoo_one_program(programs, program_id, students)
    cutoff_bonus = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    return disparity_bonus, cutoff_bonus

def add_calculations_to_bonus_values(bonus_values, bonus, disadvantaged_gender, program_id, students, programs):
    if bonus not in bonus_values.keys():
        disparity, cutoff = calculate_disparity_cutoff(bonus, disadvantaged_gender, program_id, students, programs)
        bonus_values[bonus] = {'disparity': disparity, 'cutoff': cutoff}
        
def find_ideal_bonus_gender_all_programs(cutoff_loss):
    applications = pd.read_csv('../../UniversityAdmissions/NationalData2016.csv', sep=',')
    students = prepare_data.prepare_students(applications)
    
    careers = pd.read_csv('../../UniversityAdmissions/school_data_2016.csv', sep=';')
    programs = prepare_data.prepare_universities(careers)
    
    students, programs = da.execute(students, programs)
    current_disparity = mm_measures.admissions_disparity_eoo(programs, students)
    
    bonuses = dict()

    for program_id, program in list(programs.items())[:200]:
        print('Finding ideal bonus for ' + str(program_id) + '...')
        ideal_bonus, disadvantaged_gender = find_ideal_bonus_gender(program_id, cutoff_loss, copy.deepcopy(students), programs)
        bonuses[program_id] = (ideal_bonus, disadvantaged_gender)
        if disadvantaged_gender == 'f':
            label = 'women'
        elif disadvantaged_gender == 'm':
            label = 'men'
        else:
            label = 'nobody'
        print('Ideal bonus for ' + str(program_id) + ': ' + str(ideal_bonus) + ' for ' + label)
    
    for program_id, bonus in bonuses.items():
        ideal_bonus = bonus[0]
        disadvantaged_gender = bonus[1]
        prepare_data.apply_gender_bonus_program(students, ideal_bonus, disadvantaged_gender, program_id)
    
    students, programs = da.execute(students, programs)
    improved_disparity = mm_measures.admissions_disparity_eoo(programs, students)
    
    print('Current disparity:', current_disparity, '| Improved disparity:', improved_disparity)
    
def find_ideal_bonus_gender_with_certainty(program_id, cutoff_loss, certainty, year=2016):
    applications_one_year = pd.read_csv('../../UniversityAdmissions/NationalData' + str(year) + '_enriched.csv', sep=',')
    students_one_year = prepare_data.prepare_students(applications_one_year)
    preparer = ClassificationPreparer(applications_one_year)
    applications_for_classifications_one_year = preparer.execute(applications_one_year)
    
    careers = pd.read_csv('../../UniversityAdmissions/school_data_2016.csv', sep=';')
    programs = prepare_data.prepare_universities(careers) # do this step in the other function since we don't need it before? or keep it here since we don't have the same kind of data for the other years anyway?
    
    predictor = OneTopicPerApplicationLogisticRegressionPredictor(preparer, 3)
    predictor.fit(applications_for_classifications_one_year, students_one_year)
    
    applications, average_number_of_students, average_number_of_applications = combine_previous_application_years(year)
    return find_ideal_bonus_gender_with_certainty_own_predictor(program_id, cutoff_loss, certainty, predictor, students_one_year, programs, careers, applications, average_number_of_students, average_number_of_applications)

def find_ideal_bonus_gender_with_certainty_own_predictor(program_id, cutoff_loss, certainty, predictor, students_one_year, programs, careers, applications, average_number_of_students, average_number_of_applications):
    students_one_year, programs = da.execute(students_one_year, programs)
    actual_ideal_bonus, actual_disadvantaged_gender = find_ideal_bonus_gender(program_id, cutoff_loss, students_one_year, programs)
    
    add_admitted_scores_to_programs(applications, programs)
    calculate_average_score(programs)
    
    R = 10
    ideal_bonuses = sample_ideal_bonuses(program_id, cutoff_loss, R, applications, average_number_of_students, average_number_of_applications, predictor, programs, careers)

    predicted_bonus, predicted_disadvantaged_gender = choose_bonus_with_certainty(certainty, R, ideal_bonuses)
    
    print('Predicted bonus:', predicted_bonus, predicted_disadvantaged_gender, '| Actual bonus:', actual_ideal_bonus, actual_disadvantaged_gender)
    
    return predicted_bonus, predicted_disadvantaged_gender, actual_ideal_bonus, actual_disadvantaged_gender

def combine_previous_application_years(year):
    applications = pd.DataFrame()
    average_number_of_students = 0
    average_number_of_applications = 0
    year_range = range(2004, year)
    for year in year_range:
        applications_year = pd.read_csv('../../UniversityAdmissions/NationalData' + str(year) + '_enriched.csv', sep=',')
        applications = pd.concat([applications, applications_year], ignore_index=True, sort=False)
        average_number_of_students += len(applications_year)
        for index, application in applications_year.iterrows():
            for preference in range(1, 11):
                program_id = application['career_code_' + str(preference)]
                score = application['weighted_average_score_' + str(preference)]
                if np.isfinite(program_id) and np.isfinite(score):
                    average_number_of_applications += 1
                else:
                    break
    average_number_of_students /= len(year_range)
    average_number_of_applications /= len(applications)
    
    return applications, average_number_of_students, average_number_of_applications

# =============================================================================
# def add_admitted_scores_to_programs(applications, programs):
#     for program in programs.values():
#         program['admitted_scores'] = list()
#     for index, application in applications.iterrows():
#         program_id = application['code_degree_enrolled']
#         if program_id in programs.keys():
#             programs[program_id]['admitted_scores'].append(application['admission_weighted_score'])
#     for program in programs.values():
#         program['sorted_admitted_scores'] = sorted(program['admitted_scores'])
# 
# def calculate_average_score(programs):
#     for program_id, program in programs.items():
#         if program['admitted_scores']:
#             program['average_score'] = np.mean(program['admitted_scores'])
#         else:
#             program['average_score'] = np.nan
# =============================================================================
        
def sample_ideal_bonuses(program_id, cutoff_loss, R, applications, average_number_of_students, average_number_of_applications, predictor, programs, careers):
    ideal_bonuses = []
    for r in range(R):
        M = np.random.poisson(average_number_of_students)
        random_indices = np.random.choice(len(applications),size=M)
        random_applications = applications.iloc[random_indices].copy(deep=True).reset_index()
        preferences_all_students = {}
        for index, application in random_applications.iterrows():
            preferences = predictor.predict_preferences(random_applications[index:index+1], average_number_of_applications, programs, careers)
            preferences_all_students[index] = preferences
        for choice in range(1,11):
            random_applications['career_code_' + str(choice)] = np.nan
            random_applications['weighted_average_score_' + str(choice)] = np.nan
        for index, application in random_applications.iterrows():
            preferences = preferences_all_students[index]
            for choice in range(1, len(preferences) + 1):
                random_applications.at[index, 'career_code_' + str(choice)] = preferences[choice - 1][0]
                random_applications.at[index, 'weighted_average_score_' + str(choice)] = preferences[choice - 1][1]
        sample_students = prepare_data.prepare_students(random_applications)
        ideal_bonus, disadvantaged_gender = find_ideal_bonus_gender(program_id, cutoff_loss, sample_students, programs)
        print("Sample", r, ":", disadvantaged_gender, "-", ideal_bonus)
        ideal_bonuses.append((disadvantaged_gender, ideal_bonus))
    return ideal_bonuses

def choose_bonus_with_certainty(certainty, R, ideal_bonuses):
    min_bonuses = math.ceil(certainty * R)
    female_disadvantage = [(disadvantaged_gender, bonus) for (disadvantaged_gender, bonus) in ideal_bonuses if disadvantaged_gender == 'f']
    male_disadvantage = [(disadvantaged_gender, bonus) for (disadvantaged_gender, bonus) in ideal_bonuses if disadvantaged_gender == 'm']
    if len(female_disadvantage) >= min_bonuses:
        female_disadvantage = sorted(female_disadvantage, key=lambda x: x[1])
        ideal_bonus_index = len(female_disadvantage) - min_bonuses
        predicted_disadvantaged_gender, predicted_bonus = female_disadvantage[ideal_bonus_index]
    elif len(male_disadvantage) >= min_bonuses:
        male_disadvantage = sorted(male_disadvantage, key=lambda x: x[1])
        ideal_bonus_index = len(male_disadvantage) - min_bonuses
        predicted_disadvantaged_gender, predicted_bonus = male_disadvantage[ideal_bonus_index]
    else:
        predicted_bonus = 0
        predicted_disadvantaged_gender = None
    return predicted_bonus, predicted_disadvantaged_gender
    
def which_gender_is_disadvantaged(program_id, students, programs):
    female = np.sum([student['gender'] == 'f' for student in students.values()])
    female /= len(students)
    disadvantaged_gender = None
    label = 'neither'
    admitted = programs[program_id]['accepted']
    if admitted:
        program_female = np.sum([students[student['student_id']]['gender'] == 'f' for student in admitted])
        program_female /= len(admitted)
        
        if program_female > 0.5:
            disadvantaged_gender = 'm'
            label = 'men'
        elif program_female < 0.5:
            disadvantaged_gender = 'f'
            label = 'women'
    return disadvantaged_gender, label