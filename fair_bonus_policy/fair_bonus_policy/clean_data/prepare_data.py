#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:01:13 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
import pickle
from fair_bonus_policy.utils import program_id_format
from fair_bonus_policy.config import additional_data_dir

def read_applications(file_path):
    applications = pd.read_csv(file_path, sep=',')
    applications = applications[['id', 
       'career_code_1', 'career_code_2',   'career_code_3',
       'career_code_4', 'career_code_5', 'career_code_6',
       'career_code_7', 'career_code_8', 'career_code_9', 'career_code_10',
       'weighted_average_score_1', 'weighted_average_score_2',
       'weighted_average_score_3', 'weighted_average_score_4',
       'weighted_average_score_5', 'weighted_average_score_6',
       'weighted_average_score_7', 'weighted_average_score_8',
       'weighted_average_score_9', 'weighted_average_score_10',
       'gender', 'income_decile'
       ]]
    return applications

def read_schools(file_path):
    return pd.read_csv(file_path, sep=';')


def prepare_students(applications: pd.DataFrame, with_preferences=True, only_valid_applications=True):
    students = {}
    for index, application in applications.iterrows():
        student = {}
        student['student_id'] = application['id']
        student['row_id'] = index
        student['income_category'] = application['income_decile']
        student['year'] = application['academic_year']
        student['household_size'] = application['number_family_members']
        medians = pickle.load(open(str(additional_data_dir) + '/medians', 'rb'))
        categorize_income(student, medians)
        gender_code = application['gender']
        if gender_code == 1:
            student['gender'] = 'm'
        else:
            student['gender'] = 'f'
        if with_preferences:
            student['preferences'] = {}
            for preference in range(1, 11):
                program_id = application[''.join(['career_code_', str(preference)])]
                score = application[''.join(['weighted_average_score_', str(preference)])]
                if np.isfinite(program_id) and (np.isfinite(score) or not only_valid_applications):
                    student['preferences'][preference] = {'institution_id': program_id, 'score_before_bonus': score, 'score': score}
                else:
                    break
            
        students[application['id']] = student
    return students

def prepare_students_wo_preferences(applications: pd.DataFrame):
    """Bring applications in preferred format."""
    students = {}
    for index, application in applications.iterrows():
        student = {}
        student['student_id'] = application['id']
        student['row_id'] = index
        student['income_category'] = application['income_decile']
        student['year'] = application['academic_year']
        student['household_size'] = applications['number_family_members']
        medians = pickle.load(open(str(additional_data_dir) + '/medians', 'rb'))
        categorize_income(student, medians)
        gender_code = application['gender']
        if gender_code == 1:
            student['gender'] = 'm'
        else:
            student['gender'] = 'f'
        students[application['id']] = student
    return students

def categorize_income(student: dict, medians):
    if np.isnan(student['income_category']) or np.isnan(student['household_size']):
        student['ses'] = np.nan
        return
    student_income = student['income_category'] / student['household_size']
    median = medians[student['year']]
    if student_income < median:
        student['ses'] = 'low'
    else:
        student['ses'] = 'high'
#    if student['income_category'] <= income_low:
#        student['ses'] = 'low'
#    else:
#        student['ses'] = 'high'

def apply_bonus(students: dict, bonus: int):
    """Add bonus to score of low SES students."""
    for student in students.values():
        if student['ses'] == 'low':
            for preference in student['preferences'].values():
                preference['score'] = preference['score_before_bonus'] + bonus
                
def remove_bonuses(students: dict):
    """Reset scores of all students."""
    for student in students.values():
        for preference in student['preferences'].values():
            preference['score'] = preference['score_before_bonus']
                
def apply_gender_bonus(students: dict, bonus: int, gender: str):
    """Add bonus to score of low SES students."""
    for student in students.values():
        if student['gender'] == gender:
            for preference in student['preferences'].values():
                preference['score'] = preference['score_before_bonus'] + bonus
                
def apply_gender_bonus_program(students: dict, bonus: int, gender: str, program_id: int):
    """Add bonus to score of low SES students."""
    for student in students.values():
        if student['gender'] == gender:
            for preference in student['preferences'].values():
                if program_id == preference['institution_id']:
                    preference['score'] = preference['score_before_bonus'] + bonus

def apply_bonus_to_women_program(students: dict, bonus: int, program_id: int):
    """Add bonus to score of female students."""
    if bonus >= 0:
        gender = 'f'
    else:
        gender = 'm'
    bonus = np.abs(bonus)
    for student in students.values():
        if student['gender'] == gender:
            for preference in student['preferences'].values():
                if program_id == preference['institution_id']:
                    preference['score'] = preference['score_before_bonus'] + bonus
                    
def apply_bonus_to_program(students: dict, bonus: int, program_id: int, column: str, disadvantaged: str):
    """Add bonus to score of disadvantaged students."""
    for student in students.values():
        if student[column] == disadvantaged:
            for preference in student['preferences'].values():
                if program_id == preference['institution_id']:
                    preference['score'] = preference['score_before_bonus'] + bonus

def prepare_universities(careers: pd.DataFrame, columns=None, adapt_ID_pre_2012=False):
    """Bring university data in preferred format."""
    programs = {}
    for index, career in careers.iterrows():
        program = {}
        program_id = career['CODIGO']
        if adapt_ID_pre_2012 and career['PROCESO'] <= 2011:
            program_id = program_id_format.program_id_format_2012(program_id)
        program['institution_id'] = program_id
        program['row_id'] = index
        program['quota'] = career['VACANTE_1SEM']
        program['minimum_language_math'] = career['MINIMO_LEN_MAT']
        program['choose_between_social_and_natural_sciences'] = career['HRIA_CS_ALTERNATIVA']
        program['minimum_score'] = career['MINIMO_PONDERADO']
        if 'ranking_score' in career:
            program['ranking_score'] = career['ranking_score']
        if columns is not None:
            for column in columns:
                if column in careers.columns:
                    program[column] = career[column]
        program['preferences'] = []
        programs[program_id] = program
    return programs

def add_weights_to_programs(careers, programs):
    """Adds weights used in the score function to the programs."""
    for index, career in careers.iterrows():
        if career['HRIA_CS_ALTERNATIVA'] == 'SI':
            weights = [career['%_NOTAS'], career['%_RANK'], career['%_LENG'], career['%_MATE'], career['%_HYCS']]
        else:
            weights = [career['%_NOTAS'], career['%_RANK'], career['%_LENG'], career['%_MATE'], career['%_HYCS'], career['%_CIEN']]
        program = programs[career['CODIGO']]
        program['weights'] = weights