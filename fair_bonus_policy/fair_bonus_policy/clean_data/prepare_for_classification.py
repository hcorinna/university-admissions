#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:55:15 2019

@author: hertweck
"""

import pandas as pd
import numpy as np
from scipy import stats

def execute(applications):
    applications = select_features(applications)
    applications = one_hot_encoding(applications)
    fill_nans(applications)
    applications = unite_sciences_scores(applications)
    applications = z_score_normalization(applications)
    return applications

def select_features(applications):
    applications = applications[['nationality', 'gender', #'highschool_code',
       'geo_highschool_region', 'geo_highschool_state',
       #'geo_highschool_municipality',
       'regime_educational_establishment',
       'graduation_year_media', 'birth_year', 'civil_status', 'paid_work',
       'work_hours', 'number_family_members', 'working_people_household',
       'head_household', 'primary_study_funds',
       'household_members_prebasic_education',
       'household_members_basic_education',
       'household_members_mediaIII_education',
       'household_members_mediaIV_education',
       'household_members_higher_education',
       'household_members_other_education', 'income_decile', 'health_coverage',
       'education_father', 'education_mother', 'occupation_status_father',
       'occupation_status_mother', 'father_working_organisation',
       'mother_working_organisation', 'main_occupation_father',
       'main_occupation_mother', 'type_economic_activity_father',
       'type_economic_activity_mother', 'address_region', 'address_province',
       #'address_comune', #'code_previous_university_if_applicable',
       #'admission_process',
       'education_status_student_graduation',
       'highschool_branch_type', 'type_high_school', 'GPA_HighSchool', 'nem',
       'nem_rank', 'psu_lang', 'psu_mat', 'social_sciences_test_scores',
       'psu_cie', #'application_situation',
       'pre_selected_scholarship',
       #'frashman_type',
       ]]
    return applications

def one_hot_encoding(applications):
    applications = pd.get_dummies(data=applications, columns=['nationality', 'gender', #'highschool_code',
       'geo_highschool_region', 'geo_highschool_state',
       #'geo_highschool_municipality',
       'regime_educational_establishment',
       'civil_status', 'paid_work',
       'head_household', 'primary_study_funds',
       'health_coverage',
       'occupation_status_father',
       'occupation_status_mother', 'father_working_organisation',
       'mother_working_organisation', 'main_occupation_father',
       'main_occupation_mother', 'type_economic_activity_father',
       'type_economic_activity_mother', 'address_region', 'address_province',
       #'address_comune', #'code_previous_university_if_applicable',
       #'admission_process',
       'education_status_student_graduation',
       'highschool_branch_type', 'type_high_school', #'application_situation',
       'pre_selected_scholarship',
       #'frashman_type'
       ])
    return applications

def fill_nans(applications):
    applications["number_family_members"].fillna(applications["number_family_members"].median(skipna=True), inplace=True)
    applications["education_father"].fillna(applications["education_father"].median(skipna=True), inplace=True)
    applications["education_mother"].fillna(applications["education_mother"].median(skipna=True), inplace=True)
    applications["GPA_HighSchool"].fillna(applications["GPA_HighSchool"].median(skipna=True), inplace=True)
    applications["nem"].fillna(applications["nem"].median(skipna=True), inplace=True)
    applications["nem_rank"].fillna(applications["nem_rank"].median(skipna=True), inplace=True)
    
def unite_sciences_scores(applications):
    extra_test = []
    for social, natural in zip(applications['social_sciences_test_scores'], applications['psu_cie']):
        if np.isnan(social):
            extra_test.append(natural)
        elif np.isnan(natural):
            extra_test.append(social)
        else:
            extra_test.append(max(social, natural))
    applications['extra_test'] = extra_test
    applications = applications.drop(['social_sciences_test_scores', 'psu_cie'], axis=1)
    return applications

def z_score_normalization(applications):
    zscore_columns = ['graduation_year_media', 'birth_year', 'work_hours',
          'number_family_members', 'working_people_household',
          'household_members_prebasic_education', 'household_members_basic_education', 'household_members_mediaIII_education', 'household_members_mediaIV_education', 'household_members_higher_education', 'household_members_other_education',
          'income_decile',
          'education_father', 'education_mother',
          'GPA_HighSchool', 'nem', 'nem_rank', 'psu_lang', 'psu_mat', 'extra_test']
    applications[zscore_columns] = stats.zscore(applications[zscore_columns])
    return applications
    
