#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 30 14:23:15 2019

@author: hertweck
"""

import unittest
from pandas import DataFrame
import numpy as np
import increase_social_inclusion as isi
import prepare_data as prep
from Predictor import Predictor
np.random.seed(42)

class TestPredictor(Predictor): 
  
    def predict_preferences(self, application, average_number_of_applications, programs, careers): 
        N = average_number_of_applications
        preferences = []
        for n in range(N):
            programs_left = [key for key in programs.keys() if key not in [preference[0] for preference in preferences]]
            program_probabilities = [(program_id, 1/len(programs_left)) for program_id in programs_left]
            program_id = np.random.choice([pp[0] for pp in program_probabilities], p=[pp[1] for pp in program_probabilities])
            score = 800 - application['id'] * 10
            program_quality = program_quality = programs[program_id]['average_score']
            preferences.append((program_id, score, program_quality))
        preferences = sorted(preferences, key=lambda x: x[2])
        preferences = [(preference[0],preference[1]) for preference in preferences]
        return preferences

def simple_setup_applications(applications):
    ids = []
    career_codes = [[],[],[],[],[],[],[],[],[],[]]
    weighted_average_scores = [[],[],[],[],[],[],[],[],[],[]]
    gender = []
    for sid, attributes in applications.items():
        ids.append(sid)
        gender.append(attributes[0])
        preferences = attributes[1]
        for i in range(10):
            uid = np.nan
            score = np.nan
            if i < len(preferences):
                uid = preferences[i]['uid']
                score = preferences[i]['score']
            career_codes[i].append(uid)
            weighted_average_scores[i].append(score)
        
    applications = {'id': ids,
        'career_code_1': career_codes[0],
        'career_code_2': career_codes[1],
        'career_code_3': career_codes[2],
        'career_code_4': career_codes[3],
        'career_code_5': career_codes[4],
        'career_code_6': career_codes[5],
        'career_code_7': career_codes[6],
        'career_code_8': career_codes[7],
        'career_code_9': career_codes[8],
        'career_code_10': career_codes[9],
        'weighted_average_score_1': weighted_average_scores[0],
        'weighted_average_score_2': weighted_average_scores[1],
        'weighted_average_score_3': weighted_average_scores[2],
        'weighted_average_score_4': weighted_average_scores[3],
        'weighted_average_score_5': weighted_average_scores[4],
        'weighted_average_score_6': weighted_average_scores[5],
        'weighted_average_score_7': weighted_average_scores[6],
        'weighted_average_score_8': weighted_average_scores[7],
        'weighted_average_score_9': weighted_average_scores[8],
        'weighted_average_score_10': weighted_average_scores[9],
        'income_decile': 10,
        'gender': gender
    }
    
    applications = DataFrame(applications,columns= ['id', 
       'career_code_1', 'career_code_2',   'career_code_3',
       'career_code_4', 'career_code_5', 'career_code_6',
       'career_code_7', 'career_code_8', 'career_code_9', 'career_code_10',
       'weighted_average_score_1', 'weighted_average_score_2',
       'weighted_average_score_3', 'weighted_average_score_4',
       'weighted_average_score_5', 'weighted_average_score_6',
       'weighted_average_score_7', 'weighted_average_score_8',
       'weighted_average_score_9', 'weighted_average_score_10',
       'income_decile', 'gender'])
    
    students = prep.prepare_students(applications)

    return students


class TestIncreaseSocialInclusion(unittest.TestCase):
    
    def test_no_women_zoom(self):
        applications = {1: (1, [{'uid': 1, 'score': 800}]), 2: (1, [{'uid': 1, 'score': 785}]), 3: (1, [{'uid': 1, 'score': 780}])}
        students = simple_setup_applications(applications)
        
        careers = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        ideal_bonus, disadvantaged_gender = isi.find_ideal_bonus_gender(1, 2, students, programs)
        self.assertEqual(ideal_bonus, 0)
        self.assertEqual(disadvantaged_gender, 'f')
        
    def test_no_men_zoom(self):
        applications = {1: (2, [{'uid': 1, 'score': 800}]), 2: (2, [{'uid': 1, 'score': 785}]), 3: (2, [{'uid': 1, 'score': 780}])}
        students = simple_setup_applications(applications)
        
        careers = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        ideal_bonus, disadvantaged_gender = isi.find_ideal_bonus_gender(1, 2, students, programs)
        self.assertEqual(ideal_bonus, 0)
        self.assertEqual(disadvantaged_gender, 'm')
    
    def test_2_men_1_woman_zoom(self):
        applications = {1: (1, [{'uid': 1, 'score': 800}]), 2: (1, [{'uid': 1, 'score': 785}]), 3: (2, [{'uid': 1, 'score': 780}])}
        students = simple_setup_applications(applications)
        
        careers = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        ideal_bonus, disadvantaged_gender = isi.find_ideal_bonus_gender(1, 50, students, programs)
        self.assertEqual(ideal_bonus, 6)
        self.assertEqual(disadvantaged_gender, 'f')
        
    def test_2_men_1_woman_constraint_loss_zoom(self):
        applications = {1: (1, [{'uid': 1, 'score': 800}]), 2: (1, [{'uid': 1, 'score': 785}]), 3: (2, [{'uid': 1, 'score': 780}])}
        students = simple_setup_applications(applications)
        
        careers = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        ideal_bonus, disadvantaged_gender = isi.find_ideal_bonus_gender(1, 2, students, programs)
        self.assertEqual(ideal_bonus, 0)
        self.assertEqual(disadvantaged_gender, 'f')
        
        
    def test_2_men_2_women_zoom(self):
        applications = {1: (1, [{'uid': 1, 'score': 800}]), 2: (2, [{'uid': 1, 'score': 790}]), 3: (1, [{'uid': 1, 'score': 780}]), 4: (2, [{'uid': 1, 'score': 770}])}
        students = simple_setup_applications(applications)
        
        careers = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        ideal_bonus, disadvantaged_gender = isi.find_ideal_bonus_gender(1, 50, students, programs)
        self.assertEqual(ideal_bonus, 0)
        self.assertEqual(disadvantaged_gender, None)
        
    def test_find_ideal_bonus_gender_with_certainty_own_predictor_example(self):
        applications_one_year = {1: (1, [{'uid': 1, 'score': 800}]), 2: (1, [{'uid': 1, 'score': 785}]), 3: (1, [{'uid': 1, 'score': 780}])}
        students_one_year = simple_setup_applications(applications_one_year)
        
        applications = {'id': [4, 5, 6, 7, 8, 9, 10],
            'career_code_1': [1,1,1,1,2,3,1],
            'career_code_2': [np.nan,np.nan,np.nan,2,1,2,3],
            'career_code_3': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_4': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_5': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_6': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_7': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_8': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_9': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'career_code_10': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_1': [800,790,780,790,780,800,770],
            'weighted_average_score_2': [np.nan,np.nan,np.nan,790,780,800,770],
            'weighted_average_score_3': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_4': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_5': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_6': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_7': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_8': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_9': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'weighted_average_score_10': [np.nan,np.nan,np.nan,np.nan,np.nan,np.nan,np.nan],
            'income_decile': 10,
            'gender': [1,1,1,1,1,1,1],
            'code_degree_enrolled':[1,1,1,2,1,3,3],
            'admission_weighted_score':[800,790,780,790,780,800,770]
        }
        applications = DataFrame(applications,columns= ['id', 
           'career_code_1', 'career_code_2',   'career_code_3',
           'career_code_4', 'career_code_5', 'career_code_6',
           'career_code_7', 'career_code_8', 'career_code_9', 'career_code_10',
           'weighted_average_score_1', 'weighted_average_score_2',
           'weighted_average_score_3', 'weighted_average_score_4',
           'weighted_average_score_5', 'weighted_average_score_6',
           'weighted_average_score_7', 'weighted_average_score_8',
           'weighted_average_score_9', 'weighted_average_score_10',
           'income_decile', 'gender',
           'code_degree_enrolled','admission_weighted_score'])
        
        careers = {'CODIGO': [1,2,3],
                      'VACANTE_1SEM': [1,1,1]
        }
        careers = DataFrame(careers,columns= ['CODIGO','VACANTE_1SEM'])
        programs = prep.prepare_universities(careers)
        
        predictor = TestPredictor()
        
        predicted_bonus, predicted_disadvantaged_gender, actual_ideal_bonus, actual_disadvantaged_gender = isi.find_ideal_bonus_gender_with_certainty_own_predictor(1, 10, 0.6, predictor, students_one_year, programs, careers, applications, 5, 2)
        self.assertEqual(actual_ideal_bonus, 0)
        self.assertEqual(actual_disadvantaged_gender, 'f')
        self.assertEqual(predicted_bonus, 0)
        self.assertEqual(predicted_disadvantaged_gender, 'f')
        
if __name__ == '__main__':
    unittest.main()