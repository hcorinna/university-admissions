#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  9 14:55:15 2019

@author: hertweck
"""

import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

class ClassificationPreparer:
    
    def  __init__(self, applications):
        self.social_sciences = applications["social_sciences_test_scores"].median(skipna=True)
        self.natural_sciences = applications["psu_cie"].median(skipna=True)
        
        zscore_columns = ['graduation_year_media', 'birth_year', 'work_hours',
              'number_family_members', 'working_people_household',
              'household_members_prebasic_education', 'household_members_basic_education', 'household_members_mediaIII_education', 'household_members_mediaIV_education', 'household_members_higher_education', 'household_members_other_education',
              'income_decile',
              'education_father', 'education_mother',
              'GPA_HighSchool', 'nem', 'nem_rank', 'psu_lang', 'psu_mat', 'extra_test']
        categorical_columns = ['nationality', 'gender', #'highschool_code',
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
           ]
        
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('sca', StandardScaler())])
        
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('ohe', OneHotEncoder(categories='auto', handle_unknown='ignore'))])
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, zscore_columns),
                ('cat', categorical_transformer, categorical_columns)])
        
        self.clf = Pipeline(steps=[('preprocessor', preprocessor)])
        applications = self.select_features(applications)
        applications = self.unite_sciences_scores(applications)
        self.clf.fit(applications)

    def execute(self, applications):
        applications = self.select_features(applications)
        applications = self.unite_sciences_scores(applications)
        applications = self.clf.transform(applications)
        return applications

    def select_features(self, applications):
        applications = applications.reindex(columns=['nationality', 'gender', #'highschool_code',
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
           ])
        return applications
    
    def unite_sciences_scores(self, applications):
        if 'extra_test' in applications.columns:
            return applications
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