#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 21:37:09 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
from fair_bonus_policy.clean_data import prepare_data
from fair_bonus_policy.measures import mm_measures
from fair_bonus_policy.plotting import evaluate_matching as em
from fair_bonus_policy.find_ideal_bonus import increase_social_inclusion as isi, social_inclusion_optimization as sio
from fair_bonus_policy.matching_algorithms import deferred_acceptance as da
from fair_bonus_policy.config import data_dir, career_data_path, additional_data_dir

def plot_ideal_bonus_gender(program_id, cutoff_loss):
    applications = pd.read_csv(str(data_dir) + '/NationalData2016.csv', sep=',')
    students = prepare_data.prepare_students(applications)
    
    schooldata = pd.read_csv(career_data_path, sep=';')
    programs = prepare_data.prepare_universities(schooldata)

    students, programs = da.execute(students, programs)
    cutoff_no_bonus = mm_measures.lowest_score(program_id, programs, with_bonus=False)
    disadvantaged_gender, label = isi.which_gender_is_disadvantaged(program_id, students, programs)
    
    bonus_values = np.linspace(-20, 80, 101)
    utility, disparity, spd, di, cutoff = em.calculate_disparity_utility_one_program(program_id, da.execute, students, programs, bonus_values, disadvantaged_gender)
    title = 'Deferred Acceptance (DA) with bonus for ' + label + ' applying to program ' + str(program_id)
    em.plot_utility(title, bonus_values, utility)
    em.plot_cutoff(title, bonus_values, cutoff)
    em.plot_disparity(title, bonus_values, disparity)
    min_cutoff_constraint = cutoff_no_bonus - cutoff_loss
    ideal_bonus, min_disparity = sio.minimize_disparity_for_cutoff(min_cutoff_constraint, cutoff, disparity, bonus_values)
    print('The ideal bonus is ' + str(ideal_bonus) + ' and should be applied to ' + label + ' as it does not decrease the cutoff by more than the required ' + str(cutoff_loss) + ' points and minimizes the disparity to ' + str(min_disparity) + '.')
    return ideal_bonus, disadvantaged_gender

def plot_statistical_parity_year(program_id, year, column, disadvantaged, _lambda=5, programs=None):
    applications = pd.read_csv(str(data_dir) + '/NationalData' + str(year) + '_enriched.csv')
    students = prepare_data.prepare_students(applications, True)
    if programs is None:
        careers = pd.read_csv(str(additional_data_dir) + '/career_data_' + str(year) + '.csv')
        programs = prepare_data.prepare_universities(careers)
    students, programs = da.execute(students, programs)
    
    bonus_values = np.linspace(-120, 120, 241)
    utility, spd, di, cutoff = em.calculate_statistical_parity_one_program(program_id, da.execute, students, programs, bonus_values, column, disadvantaged)
    title = 'Deferred Acceptance (DA) with bonus for ' + disadvantaged + ' applying to program ' + str(program_id)
    em.plot_utility(title, bonus_values, utility)
    em.plot_cutoff(title, bonus_values, cutoff)
    em.plot_statistical_parity_difference(title, bonus_values, spd)
    em.plot_disparate_impact(title, bonus_values, di)
    zero_index, = np.where(bonus_values == 0)
    zero_index = zero_index[0]
    #utility_difference = utility[zero_index] - np.array(utility)
    em.plot_objective_function(title, bonus_values, utility, spd, _lambda)