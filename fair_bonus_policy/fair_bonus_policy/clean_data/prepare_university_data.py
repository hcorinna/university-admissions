#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 09:43:58 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
import unicodedata
from sklearn.linear_model import LogisticRegression
from fair_bonus_policy.config import additional_data_dir, career_data_path

def standardize_university_names(schooldata):
    schooldata['UNIVERSIDAD'] = [clear_university_name(name) for name in schooldata['UNIVERSIDAD']]

def add_university_ids(schooldata, universities):
    """Add the university to which each student was accepted to the set of applicants"""
    schooldata['university_id'] = np.nan
    for university_id, university in universities.items():
        schooldata.at[schooldata['UNIVERSIDAD'] == university['name'], 'university_id'] = university_id

def create_universities_dict():
    university_ids = pd.read_csv(str(additional_data_dir) + '/university_ids.csv', sep=';', header=None)
    university_ids.columns = ['university_name', 'university_id']
    universities = dict()
    for index, university in university_ids.iterrows():
        university_name = clear_university_name(university['university_name'])
        universities[university['university_id']] = {'name': university_name, 'programs': dict()}
    return universities

def strip_accents(s):
    return ''.join(c for c in unicodedata.normalize('NFD', s) if unicodedata.category(c) != 'Mn')

def clear_university_name(university_name):
    university_name = university_name.lower()
    university_name = university_name.strip()
    university_name = university_name.replace('-', '')
    university_name = strip_accents(university_name)
    return university_name

def add_university_rankings(universities):
    university_ranking = pd.read_csv(str(additional_data_dir) + '/university_ranking.csv', sep=';', header=None)
    university_ranking.columns = ['ranking', 'university_name', 'location']
    university_ranking['university_name'] = [clear_university_name(name) for name in university_ranking['university_name']]
    for university_id, university in universities.items():
        ranking = university_ranking[university_ranking['university_name'] == university['name']]['ranking'].to_numpy()[0]
        university['ranking'] = ranking

def nest_programs_in_universities(schooldata, universities):
    all_programs = dict()
    for index, school in schooldata.iterrows():
        program = dict()
        program['row_id'] = index
        program['program_id'] = school['CODIGO']
        program['quota'] = school['VACANTE_1SEM']
        program['university_id'] = school['university_id']
        program['admitted'] = dict()
        program['applicants'] = dict()
        university = universities[school['university_id']]
        university['programs'][school['CODIGO']] = program
        all_programs[school['CODIGO']] = program
    return all_programs

def add_students_to_programs(applications, programs):
    for index, application in applications.iterrows():
        program = application['code_degree_enrolled']
        programs[program]['admitted'][application['id']] = index

def calculate_average_scores(universities, applications):
    for university_id, university in universities.items():
        indices_university = []
        for program_id, program in university['programs'].items():
            indices_program = list(program['admitted'].values())
            indices_university.extend(indices_program)
            
            admitted_program = applications.iloc[indices_program]
            program['average_score'] = np.mean(admitted_program['admission_weighted_score'])
        
        admitted_university = applications.iloc[indices_university]
        university['average_score'] = np.mean(admitted_university['admission_weighted_score'])
        
def sort_dict_with_nans(data, sort_by, reverse=True):
    to_sort = {key:value for (key,value) in data.items() if np.isfinite(value[sort_by])}
    nan_data = {key:value for (key,value) in data.items() if np.isnan(value[sort_by])}
    sorted_data = sorted(to_sort.items(), reverse=reverse, key=lambda kv: kv[1][sort_by])
    sorted_data.extend([(program_id, program) for program_id, program in nan_data.items()])
    return sorted_data

def create_programs_dict():
    """Create dict with empty dict for each program where the keys are the program IDs."""
    careers = pd.read_csv(career_data_path, sep=';')
    programs = {}
    for index, career in careers.iterrows():
        program = {}
        programs[career['CODIGO']] = program
    return programs

def add_admitted_scores_to_programs(applications, programs, year_range):
    for program in programs.values():
        program['admitted_scores'] = {}
        program['sorted_admitted_scores'] = []
    for application_year in year_range:
        applications_in_year = applications.loc[application_year]
        # if standardize:
            # applications_in_year[['admission_weighted_score']] = np.array(applications_in_year[['admission_weighted_score']].apply(zscore))
        for index, application in applications_in_year.iterrows():
            program_id = application['code_degree_enrolled']
            if program_id in programs:
                programs[program_id]['admitted_scores'][application_year] = programs[program_id]['admitted_scores'].get(application_year, [])
                programs[program_id]['admitted_scores'][application_year].append(application['admission_weighted_score'])
                programs[program_id]['sorted_admitted_scores'].append(application['admission_weighted_score'])
    for program in programs.values():
        program['sorted_admitted_scores'] = sorted(program['sorted_admitted_scores'])

def add_applicants_to_programs(applications, programs, year):
    for program in programs.values():
        program['applicants'] = {year: {}}
    for index, application in applications.iterrows():
        for preference in range(1,11):
            program_id = application['career_code_' + str(preference)]
            score = application['weighted_average_score_' + str(preference)]
            if np.isfinite(program_id) and np.isfinite(score):
                programs[program_id]['applicants'][year][application['id']] = index
            if program_id == application['code_degree_enrolled']:
                break
                
def add_applicants_with_results_to_programs(applications, programs, year_range):
    for program in programs.values():
        program['application_scores_results'] = {}
        for application_year in year_range:
            program['application_scores_results'][application_year] = []
    for application_year in year_range:
        applications_in_year = applications.loc[application_year]
        for index, application in applications_in_year.iterrows():
            for preference in range(1,11):
                program_id = application['career_code_' + str(preference)]
                if program_id in programs:
                    score = application['weighted_average_score_' + str(preference)]
                    if program_id == application['code_degree_enrolled']:
                        programs[program_id]['application_scores_results'][application_year].append((score,1))
                        break
                    if np.isfinite(program_id) and np.isfinite(score) and program_id in programs.keys():
                        programs[program_id]['application_scores_results'][application_year].append((score,0))
                
def get_admission_logistic_regression(programs, year_range):
    get_probabilities = {}
    for program_id, program in programs.items():
        scores_results = []
        for application_year in year_range:
            scores_results.extend(program['application_scores_results'][application_year])
        clf = None
        if len(scores_results) > 0:
            X = [score for (score, result) in scores_results if np.isfinite(score)] + [0]
            y = [result for (score, result) in scores_results if np.isfinite(score)] + [0]
            X = np.array(X)
            X = X[:, np.newaxis]
            clf = LogisticRegression(random_state=0, C=1e5, solver='lbfgs')
            clf.fit(X, y)
        get_probabilities[program_id] = clf
    return get_probabilities

def calculate_average_score_previous_years(programs):
    for program_id, program in programs.items():
        program['average_score'] = dict()
        scores = []
        for application_year, admitted_scores in program['admitted_scores'].items():
            scores.extend(admitted_scores)
            program['average_score'][application_year] = np.mean(scores)    
            
def calculate_averaged_score_year_range(programs, year_range):
    for program_id, program in programs.items():
        scores = []
        for year in year_range:
            program['admitted_scores'][year] = program['admitted_scores'].get(year, [])
            admitted_scores = program['admitted_scores'][year]
            scores.extend(admitted_scores)
        if len(scores) > 0:
            program['averaged_score'] = np.mean(scores)
        else:
            program['averaged_score'] = np.nan