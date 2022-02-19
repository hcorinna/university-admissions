#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 14:57:28 2020

@author: hertweck
"""

import numpy as np
import pandas as pd
from fair_bonus_policy.clean_data import prepare_data, clean_data
from fair_bonus_policy.config import data_dir, additional_data_dir


def create_learning_to_rank_data_n_preferences(lower_limit, upper_limit, program_per_student=100):
    applications, students, careers = preprocessing(lower_limit, upper_limit)
    
    rows_list = []
    career_codes = careers['CODIGO'].tolist()
    careers_dict = careers.to_dict('records')
    for application_row in applications.to_dict('records'):
        preferences = students[application_row['id']]['preferences']
        preferences_IDs = [preference['institution_id'] for preference in preferences.values()]
        sample_codes = list(set(career_codes).difference(preferences_IDs))
        program_IDs = np.random.choice(a=sample_codes, size=program_per_student-len(preferences_IDs)).tolist()
        program_IDs.extend(preferences_IDs)
        program_rows = [career_row for career_row in careers_dict if career_row['CODIGO'] in program_IDs]
        for program_row in program_rows:
            new_row = application_row.copy()
            new_row.update(program_row)
            program_id = program_row['CODIGO']
            if program_id in preferences_IDs:
                choice = preferences_IDs.index(program_id)
                score = 10 - choice
            else:
                score = 0
            new_row['score'] = score
            rows_list.append(new_row)
    applications_careers = pd.DataFrame(rows_list)
    return applications_careers

def create_learning_to_rank_data_all_preferences(lower_limit, upper_limit):
    applications, students, careers = preprocessing(lower_limit, upper_limit)
    
    applications['score'] = 0
    careers['score'] = 0
    applications_careers = applications[lower_limit:upper_limit].merge(careers, how='outer')
    
    student_ids = applications['id'][lower_limit:upper_limit].tolist()
    add_relevance_scores(applications_careers, students, student_ids)
    
    return applications_careers

def preprocessing(lower_limit, upper_limit):
    applications, students = get_application_data(lower_limit, upper_limit)
    applications = reduce_size_dtype(applications)
    careers = pd.read_csv(str(additional_data_dir) + '/careers_2016_cleaned.csv')
    careers = reduce_size_dtype(careers)
    careers = reduce_size_descriptor_columns(careers)
    return applications, students, careers

def get_application_data(lower_limit, upper_limit):
    applications = pd.read_csv(str(data_dir) + '/NationalData2016_enriched.csv')
    applications = applications[lower_limit:upper_limit]
    students = prepare_data.prepare_students(applications, with_preferences=True, only_valid_applications=True)
    applications = clean_data.execute(applications, drop_id=False)
    return applications, students

def reduce_size_dtype(df):
    int64_indices = [i for i, x in enumerate(df.dtypes) if x == 'int64']
    int64_columns = df.columns[int64_indices]
    df[int64_columns] = df[int64_columns].astype('int32')
    float64_indices = [i for i, x in enumerate(df.dtypes) if x == 'float64']
    float64_columns = df.columns[float64_indices]
    df[float64_columns] = df[float64_columns].astype('float32')
    return df

def reduce_size_descriptor_columns(careers):
    descriptor_columns = [x for i, x in enumerate(careers.columns) if x.startswith('descriptor_')]
    careers[descriptor_columns] = careers[descriptor_columns].astype('uint8')
    return careers

def add_relevance_scores(applications_careers, students, student_ids):
    for student_id, student in students.items():
        if student_id in student_ids:
            for choice, preference in student['preferences'].items():
                program_id = preference['institution_id']
                applications_careers.loc[(applications_careers['id'] == student_id) & (applications_careers['CODIGO'] == program_id), 'score'] = 10 - (choice - 1)
    return applications_careers