#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 26 20:49:40 2019

@author: hertweck
"""

import numpy as np
import pandas as pd
from scipy.stats import zscore

def execute(applications, year_range=None, drop_id=True):
    
    applications = drop_columns(applications, drop_id)
    
    applications = add_tests_taken(applications)
    
    applications = impute(applications)
    
    applications = scale(applications, year_range)
        
    applications = encode(applications)
    
    return applications

def drop_columns(applications, drop_id):
    if drop_id:
        applications = applications.drop('id', axis=1)
    return applications.drop([
        'academic_year', 'code_degree_enrolled', 'university_choosen', 'frashman_type', 'admitted_not_admitted', 'application_situation', 'paid_work', 'highschool_code',
        'career_code_1', 'career_code_2', 'career_code_3', 'career_code_4', 'career_code_5', 'career_code_6', 'career_code_7', 'career_code_8', 'career_code_9', 'career_code_10',
        'preference_situation_1', 'preference_situation_2', 'preference_situation_3', 'preference_situation_4', 'preference_situation_5', 'preference_situation_6', 'preference_situation_7', 'preference_situation_8', 'preference_situation_9', 'preference_situation_10',
        'weighted_average_score_1', 'weighted_average_score_2', 'weighted_average_score_3', 'weighted_average_score_4', 'weighted_average_score_5', 'weighted_average_score_6', 'weighted_average_score_7', 'weighted_average_score_8', 'weighted_average_score_9', 'weighted_average_score_10',
        'scores_used_1', 'scores_used_2', 'scores_used_3', 'scores_used_4', 'scores_used_5', 'scores_used_6', 'scores_used_7', 'scores_used_8', 'scores_used_9', 'scores_used_10',
        'pre_selected_scholarship', 'admission_process', 'admission_weighted_score', 'winning_preference',
        'percentage_admitted', 'percentage_admitted.var',
           'total_students_admitted', 'GPA_HighSchool.mean', 'GPA_HighSchool.var',
           'GPA_HighSchool.sum', 'highschool_branch_type.mean',
           'income_decile.mean', 'income_decile.var', 'income_decile.sum',
           'nem.mean', 'nem.var', 'nem.sum', 'psu_cie.mean', 'psu_cie.var',
           'psu_cie.sum', 'psu_lang.mean', 'psu_lang.var', 'psu_lang.sum',
           'psu_mat.mean', 'psu_mat.var', 'psu_mat.sum',
           'social_sciences_test_scores.mean', 'social_sciences_test_scores.var',
           'social_sciences_test_scores.sum', 'geo_highschool_municipality.mean'
    ], axis=1)

def add_tests_taken(applications):
    tests_taken = []
    for social, natural in zip(applications['social_sciences_test_scores'], applications['psu_cie']):
        if np.isnan(social):
            tests_taken.append('C')
        elif np.isnan(natural):
            tests_taken.append('S')
        else:
            tests_taken.append('CS')
    applications['tests_taken'] = tests_taken
    return applications
        
def impute(applications):
    applications['has_been_accepted_before'] = np.where(applications['code_previous_university_if_applicable'].isnull(), 0, 1)
    applications = applications.drop(['code_previous_university_if_applicable'], axis=1)
    
    median_features = ['income_decile', 'GPA_HighSchool', 'nem', 'social_sciences_test_scores', 'psu_cie', 'number_family_members']
    applications[median_features] = applications[median_features].fillna(applications[median_features].median())
    
    applications['nem_rank'] = applications['nem_rank'].fillna(applications['nem'])
    
    applications.loc[applications['geo_highschool_region'] == 99, 'geo_highschool_region'] = np.nan
    applications['geo_highschool_region'] = np.where(applications['geo_highschool_region'].isnull(), applications['address_region'], applications['geo_highschool_region'])
    applications['address_region'] = np.where(applications['address_region'].isnull(), applications['geo_highschool_region'], applications['address_region'])
    
    applications.loc[applications['geo_highschool_state'] == 991, 'geo_highschool_state'] = np.nan
    applications['geo_highschool_region'] = np.where(applications['geo_highschool_region'].isnull(), applications['address_province'], applications['geo_highschool_region'])
    applications['address_province'] = np.where(applications['address_province'].isnull(), applications['geo_highschool_state'], applications['address_province'])
    
    applications['geo_highschool_municipality'] = np.where(applications['geo_highschool_municipality'].isnull(), applications['address_comune'], applications['geo_highschool_municipality'])
    applications['address_comune'] = np.where(applications['address_comune'].isnull(), applications['geo_highschool_municipality'], applications['address_comune'])
    
    
    applications.loc[np.isnan(applications['education_father']), 'education_father'] = 13
    applications.loc[np.isnan(applications['education_mother']), 'education_mother'] = 13
    
    applications['main_occupation_father'] = np.where((np.isnan(applications['main_occupation_father'])) & ((applications['father_working_organisation'] == 6) | (applications['occupation_status_father'] == 12) | (applications['type_economic_activity_father'] == 12)), 6, applications['main_occupation_father'])
    applications['father_working_organisation'] = np.where((np.isnan(applications['father_working_organisation'])) & ((applications['main_occupation_father'] == 6) | (applications['occupation_status_father'] == 12) | (applications['type_economic_activity_father'] == 12)), 6, applications['father_working_organisation'])
    applications['occupation_status_father'] = np.where((np.isnan(applications['occupation_status_father'])) & ((applications['main_occupation_father'] == 6) | (applications['father_working_organisation'] == 6) | (applications['type_economic_activity_father'] == 12)), 12, applications['occupation_status_father'])
    applications['type_economic_activity_father'] = np.where((np.isnan(applications['type_economic_activity_father'])) & ((applications['main_occupation_father'] == 6) | (applications['father_working_organisation'] == 6) | (applications['occupation_status_father'] == 12)), 12, applications['type_economic_activity_father'])
    
    applications['occupation_status_father'] = applications['occupation_status_father'].fillna(7)
    applications['father_working_organisation'] = applications['father_working_organisation'].fillna(7)
    applications['main_occupation_father'] = applications['main_occupation_father'].fillna(13)
    applications['type_economic_activity_father'] = applications['type_economic_activity_father'].fillna(13)
    
    applications['main_occupation_mother'] = np.where((np.isnan(applications['main_occupation_mother'])) & ((applications['mother_working_organisation'] == 6) | (applications['occupation_status_mother'] == 12) | (applications['type_economic_activity_mother'] == 12)), 6, applications['main_occupation_mother'])
    applications['mother_working_organisation'] = np.where((np.isnan(applications['mother_working_organisation'])) & ((applications['main_occupation_mother'] == 6) | (applications['occupation_status_mother'] == 12) | (applications['type_economic_activity_mother'] == 12)), 6, applications['mother_working_organisation'])
    applications['occupation_status_mother'] = np.where((np.isnan(applications['occupation_status_mother'])) & ((applications['main_occupation_mother'] == 6) | (applications['mother_working_organisation'] == 6) | (applications['type_economic_activity_mother'] == 12)), 12, applications['occupation_status_mother'])
    applications['type_economic_activity_mother'] = np.where((np.isnan(applications['type_economic_activity_mother'])) & ((applications['main_occupation_mother'] == 6) | (applications['mother_working_organisation'] == 6) | (applications['occupation_status_mother'] == 12)), 12, applications['type_economic_activity_mother'])
    
    applications['occupation_status_mother'] = applications['occupation_status_mother'].fillna(7)
    applications['mother_working_organisation'] = applications['mother_working_organisation'].fillna(7)
    applications['main_occupation_mother'] = applications['main_occupation_mother'].fillna(13)
    applications['type_economic_activity_mother'] = applications['type_economic_activity_mother'].fillna(13)
    
    mode_features = [
        'address_region', 'geo_highschool_region',
        'geo_highschool_state', 'address_province',
        'geo_highschool_municipality', 'address_comune',
        'regime_educational_establishment', 'civil_status', 'head_household', 'primary_study_funds', 'health_coverage', 'type_high_school'
    ]
    applications[mode_features] = applications[mode_features].fillna(applications[mode_features].mode().iloc[0])
    
    return applications

def scale(applications, year_range):
    zscore_features = ['graduation_year_media',
           'birth_year', 'work_hours', 'number_family_members',
           'working_people_household', 
           'household_members_prebasic_education',
           'household_members_basic_education',
           'household_members_mediaIII_education',
           'household_members_mediaIV_education',
           'household_members_higher_education',
           'household_members_other_education', 'income_decile', 'GPA_HighSchool', 'nem',
           'psu_lang', 'psu_mat', 'nem_rank', 'social_sciences_test_scores', 'psu_cie']
    
    if year_range is None:
        applications[zscore_features] = np.array(applications[zscore_features].apply(zscore))
    else:
        for year in year_range:
            applications.loc[year, zscore_features] = np.array(applications.loc[year, zscore_features].apply(zscore))
    
    return applications

def encode(applications):
    applications.loc[applications['nationality'] == 2, 'nationality'] = 0
    applications.loc[applications['gender'] == 2, 'gender'] = 0
    applications = applications.astype({'gender': 'int64', 'nationality': 'int64'})
    
    one_hot_features = ['geo_highschool_region',
           'geo_highschool_state', 'geo_highschool_municipality',
            'address_region', 'address_province', 'address_comune', 
           'regime_educational_establishment', 'civil_status', 'head_household', 'primary_study_funds', 'health_coverage',
           'education_father', 'education_mother', 'occupation_status_father',
           'occupation_status_mother', 'father_working_organisation',
           'mother_working_organisation', 'main_occupation_father',
           'main_occupation_mother', 'type_economic_activity_father',
           'type_economic_activity_mother', 'education_status_student_graduation',
           'highschool_branch_type', 'type_high_school', 'tests_taken']
    
    applications = pd.get_dummies(applications, columns=one_hot_features)
    
    return applications