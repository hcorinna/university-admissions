#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Oct  9 11:28:53 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import MultiLabelBinarizer
from hyperopt import hp, tpe, Trials, fmin
import pickle
import copy

from fair_bonus_policy.clean_data import prepare_data, clean_data, prepare_university_data as pud
from fair_bonus_policy.measures import mm_measures
from fair_bonus_policy.matching_algorithms import deferred_acceptance as da
from fair_bonus_policy.utils import access_data as ad
from fair_bonus_policy.application_predictions.Predictor import DecisionTreePredictor, RandomForestPredictor, RandomForestRegressorPredictor
from fair_bonus_policy.config import data_dir, additional_data_dir

import cProfile

class SimpleObjective:
    
    def  __init__(self, program_id, column, disadvantaged, programs, pools_path, _lambda=None, minimize=True):
        """When pools have already been sampled."""
        self.program_id = program_id
        self.column = column
        self.disadvantaged = disadvantaged
        self.programs = programs
        self._lambda = _lambda
        self.minimize = minimize
        self.pools_path = pools_path
        self.evaluation_counter = 0
        
    def disparity(self, bonus):
        """Objective function to minimize"""
        if isinstance(bonus, list) and len(bonus) == 1:
            bonus = bonus[0]
        self.evaluation_counter += 1
        pool_path = self.pools_path + str(self.evaluation_counter)
        sample_students = ad.get_pool(pool_path)
        result = evaluate_bonus(bonus, sample_students, self.programs, self.program_id, self._lambda, self.column, self.disadvantaged)
        if self.minimize:
            return result
        return -result

class Objective:
    
    def  __init__(self, program_id, column, disadvantaged, applications_for_sampling, applications_for_predictions, careers, programs, average_number_of_students, predictor, _lambda=None, minimize=True, pools_path=None):
        self.program_id = program_id
        self.column = column
        self.disadvantaged = disadvantaged
        self.applications_for_sampling = applications_for_sampling
        self.applications_for_predictions = applications_for_predictions
        self.careers = careers
        self.programs = programs
        self.average_number_of_students = average_number_of_students
        self.predictor = predictor
        self._lambda = _lambda
        self.minimize = minimize
        self.pools_path = pools_path
        self.evaluation_counter = 0
        
    def disparity(self, bonus):
        """Objective function to minimize"""
        if isinstance(bonus, list) and len(bonus) == 1:
            bonus = bonus[0]
        self.evaluation_counter += 1
        sample_students = get_sample_students(self.average_number_of_students, self.applications_for_sampling, self.applications_for_predictions, self.predictor, self.evaluation_counter, self.pools_path)
        result = evaluate_bonus(bonus, sample_students, self.programs, self.program_id, self._lambda, self.column, self.disadvantaged)
        if self.minimize:
            return result
        return -result
    
class IntersectionalObjective:
    
    def  __init__(self, program_id, applications_for_sampling, applications_for_predictions, careers, programs, average_number_of_students, predictor, _lambdas=None, minimize=True, pools_path=None):
        self.program_id = program_id
        self.columns = ['gender', 'ses']
        self.disadvantaged_groups = ['f', 'low']
        self.applications_for_sampling = applications_for_sampling
        self.applications_for_predictions = applications_for_predictions
        self.careers = careers
        self.programs = programs
        self.average_number_of_students = average_number_of_students
        self.predictor = predictor
        self._lambdas = _lambdas
        self.minimize = minimize
        self.pools_path = pools_path
        self.evaluation_counter = 0
        
    def disparity(self, bonuses):
        """Objective function to minimize"""
        self.evaluation_counter += 1
        sample_students = get_sample_students(self.average_number_of_students, self.applications_for_sampling, self.applications_for_predictions, self.predictor, self.evaluation_counter, self.pools_path)
        bonuses_list = [bonuses['gender_bonus'], bonuses['ses_bonus']]
        result = evaluate_bonuses(bonuses_list, sample_students, self.programs, self.program_id, self._lambdas, self.columns, self.disadvantaged_groups)
        if self.minimize:
            return result
        return -result
    
def get_sample_students(average_number_of_students, applications_for_sampling, applications_for_predictions, predictor, evaluation_counter, pools_path=None):
    sample_students = None
    if pools_path is not None:
        pool_path = pools_path + str(evaluation_counter)
        sample_students = ad.get_pool(pool_path)
    if sample_students is None:
        sample_students = sample_applications(average_number_of_students, applications_for_sampling, applications_for_predictions, predictor)
        if pools_path is not None:
            pickle.dump(sample_students, pool_path)
    
def sample_applications(average_number_of_students, applications_for_sampling, applications_for_predictions, predictor):
    M = np.random.poisson(average_number_of_students)
    random_indices = np.random.choice(len(applications_for_sampling),size=M)
    sample_applications = applications_for_sampling.iloc[random_indices].copy(deep=True)
    sample_X = applications_for_predictions.iloc[random_indices].copy(deep=True)
    sample_students = prepare_data.prepare_students(sample_applications, False)
    
    predictor.predict_preferences(sample_X, sample_applications, sample_students)
    
    return sample_students

def evaluate_bonus(bonus, original_students, original_programs, program_id, _lambda, column, disadvantaged):
    students = copy.deepcopy(original_students)
    programs = copy.deepcopy(original_programs)
    if _lambda:
        students, programs = da.execute(students, programs)
        original_utility = mm_measures.admissions_utility_one_program(programs, program_id)
    prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    spd, di = mm_measures.statistical_parity_one_program(programs, program_id, students, column, disadvantaged)
#    print('SPD:', spd)
#    print('DI:', di)
    if _lambda:
        utility = mm_measures.admissions_utility_one_program(programs, program_id)
        objective, utility_difference, disparity = mm_measures.objective_function_point(utility, original_utility, [spd], [_lambda])
        return objective
    return np.abs(spd)

def evaluate_bonuses(bonuses, original_students, original_programs, program_id, _lambdas, columns, disadvantaged_groups):
    students = copy.deepcopy(original_students)
    programs = copy.deepcopy(original_programs)
    if _lambdas is not None and len(_lambdas) > 0:
        students, programs = da.execute(students, programs)
        original_utility = mm_measures.admissions_utility_one_program(programs, program_id)
    for bonus, column, disadvantaged in zip(bonuses, columns, disadvantaged_groups):
        prepare_data.apply_bonus_to_program(students, bonus, program_id, column, disadvantaged)
    students, programs = da.execute(students, programs)
    disparities = mm_measures.statistical_parities_one_program(programs, program_id, students, columns, disadvantaged_groups)
    spds = [spd for spd, di in disparities]
    if _lambdas is not None and len(_lambdas) > 0:
        utility = mm_measures.admissions_utility_one_program(programs, program_id)
        objective, utility_difference, disparity = mm_measures.objective_function_point(utility, original_utility, spds, _lambdas)
        return objective
    return sum(np.abs(spd) for spd in spds)
    
def setup(train_year, year_range_quality=None):
    year_range = range(2004, (train_year+1))
    all_applications = combine_applications_from_years(year_range)
    average_number_of_students = int(len(all_applications) / len(year_range))
    
    applications_for_predictions = all_applications.copy(deep=True)
    applications_for_predictions = clean_data.execute(applications_for_predictions, year_range)
    
    applications_for_sampling = all_applications.copy(deep=True)
    applications_for_sampling = clean_data.add_tests_taken(applications_for_sampling)
    applications_for_sampling = clean_data.impute(applications_for_sampling)
    
    careers = pd.read_csv(str(additional_data_dir) + '/spots_2004_2017.csv', sep=';')
    careers = careers.loc[careers['PROCESO'] == train_year]
    
    programs = prepare_data.prepare_universities(careers)
    if year_range_quality is not None:
        pud.add_admitted_scores_to_programs(all_applications, programs, year_range_quality)
        pud.calculate_averaged_score_year_range(programs, year_range_quality)
    prepare_data.add_weights_to_programs(careers, programs)
    
    return all_applications, applications_for_sampling, applications_for_predictions, careers, programs, average_number_of_students

def get_predictor(train_year, predictor_type, model, all_applications, applications_for_predictions, parameters, careers, programs):
    program_ids = careers['CODIGO'].values
    X = applications_for_predictions.loc[train_year]
    ranked = True if predictor_type == 'rf_regressor' else False
    y = get_y_for(all_applications.loc[train_year], program_ids, ranked)
    if predictor_type == 'dtree':
        predictor = DecisionTreePredictor(parameters, careers, programs, model)
    elif predictor_type == 'rf_classifier':
        predictor = RandomForestPredictor(parameters, careers, programs, model)
    elif predictor_type == 'rf_regressor':
        predictor = RandomForestRegressorPredictor(parameters, careers, programs, model)
    predictor.fit(X, y)
    return predictor

def combine_applications_from_years(year_range):
    all_applications = pd.DataFrame()
    applications_dict = {}
    for year in year_range:
        applications_year = pd.read_csv(str(data_dir) + '/NationalData' + str(year) + '_enriched.csv', sep=',')
        applications_dict[year] = applications_year
    all_applications = pd.concat(applications_dict, sort=False)
    return all_applications

def get_y_for(applications, program_ids, ranked=False):
    if ranked:
        return get_y_ranking_for(applications, program_ids)
    return get_y_binary_for(applications, program_ids)

def get_y_binary_for(applications, program_ids):
    preferences = []
    for index, application in applications.iterrows():
        student_preferences = []
        for choice in range(1,11):
            program_id = application['career_code_' + str(choice)]
            #score = application['weighted_average_score_' + str(choice)]
            if np.isnan(program_id): # or np.isnan(score):
                break
            student_preferences.append(program_id)
        preferences.append(student_preferences)
    
    mlb = MultiLabelBinarizer(classes=program_ids)
    y = mlb.fit_transform(preferences)
    return y

def get_y_ranking_for(applications, program_ids):
    y_ranking = np.zeros((len(applications), len(program_ids)))
    for index, application in applications.iterrows():
        for choice in range(1,11):
            program_id = application['career_code_' + str(choice)]
#            score = application['weighted_average_score_' + str(choice)]
            if np.isnan(program_id):# or np.isnan(score):
                break
            column, = np.where(program_ids == program_id)
            rank = 10 - (choice - 1)
            y_ranking[index, column] = rank
    return y_ranking


def run_bayesian_optimization(program_id, train_year=2016, year_range_quality=range(2013, 2016), number_of_times=1, utility_weight=0.1):
    
    parameters = {
            'type': 'dtree',
            'criterion': 'entropy',
            'max_depth': 10,
            'min_samples_split': 18
    }
    
    applications_for_sampling, applications_for_predictions, careers, programs, average_number_of_students, predictor = setup(train_year, year_range_quality, parameters)
    
    objective = Objective(program_id, applications_for_sampling, applications_for_predictions, careers, programs, average_number_of_students, predictor, utility_weight)
    
    space = hp.qnormal('bonus', 0, 20, 1)
    tpe_algo = tpe.suggest
    
    tpe_trials = Trials()
    
    tpe_best = fmin(fn=objective.disparity, space=space, algo=tpe_algo, trials=tpe_trials, max_evals=number_of_times, rstate=np.random.RandomState(50))
    
    print(tpe_best)