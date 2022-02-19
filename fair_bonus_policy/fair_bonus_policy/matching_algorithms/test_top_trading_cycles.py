n#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 13:37:00 2019

@author: hertweck
"""

import unittest
from pandas import DataFrame
import numpy as np
import prepare_data as prep
import top_trading_cycles as ttc

def simple_setup_applications(applications):
    ids = []
    career_codes = [[],[],[],[],[],[],[],[],[],[]]
    weighted_average_scores = [[],[],[],[],[],[],[],[],[],[]]
    for sid, preferences in applications.items():
        ids.append(sid)
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
        'income_decile': 10
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
       'income_decile'])
    
    students = prep.prepare_students(applications)

    return students


class TestTopTradingCycles(unittest.TestCase):
    
    def test_student_without_application_ttc(self):
        applications = {1: [{'uid': 1, 'score': 800}], 2: []}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 'none')
    
    def test_two_students_two_schools_ttc(self):
        applications = {1: [{'uid': 1, 'score': 800}], 2: [{'uid': 2, 'score': 600}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 2)
        
    def test_two_students_one_school_ttc(self):
        applications = {1: [{'uid': 1, 'score': 800}], 2: [{'uid': 1, 'score': 600}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1],
                      'VACANTE_1SEM': [1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 'none')
        
    def test_two_students_two_seats_ttc(self):
        applications = {1: [{'uid': 1, 'score': 800}], 2: [{'uid': 1, 'score': 600}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1],
                      'VACANTE_1SEM': [2]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 1)
    
    def test_score_over_preference_when_better_student_exists_ttc(self):
        applications = {1: [{'uid': 1, 'score': 800}], 2: [{'uid': 2, 'score': 600}], 3: [{'uid': 1, 'score': 700}, {'uid': 2, 'score': 700}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 'none')
        self.assertEqual(students[3]['status'], 2)
    
    def test_preference_over_score_ttc(self):
        applications = {1: [{'uid': 1, 'score': 700}, {'uid': 2, 'score': 800}], 2: [{'uid': 2, 'score': 700}, {'uid': 1, 'score': 800}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 1)
        self.assertEqual(students[2]['status'], 2)
    
    def test_switched_preference_ttc(self):
        applications = {
                1: [{'uid': 1, 'score': 600}, {'uid': 2, 'score': 750}],
                2: [{'uid': 1, 'score': 700}, {'uid': 2, 'score': 800}],
                3: [{'uid': 1, 'score': 850}, {'uid': 2, 'score': 850}],
                4: [{'uid': 2, 'score': 700}, {'uid': 1, 'score': 750}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [2,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 'none')
        self.assertEqual(students[2]['status'], 1)
        self.assertEqual(students[3]['status'], 1)
        self.assertEqual(students[4]['status'], 2)
        
    def test_two_cycles_ttc(self):
        applications = {
                1: [{'uid': 1, 'score': 700}],
                2: [{'uid': 1, 'score': 700}, {'uid': 2, 'score': 800}, {'uid': 3, 'score': 800}],
                3: [{'uid': 2, 'score': 700}, {'uid': 1, 'score': 800}],
                4: [{'uid': 3, 'score': 700}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2,3],
                      'VACANTE_1SEM': [1,1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 'none')
        self.assertEqual(students[2]['status'], 1)
        self.assertEqual(students[3]['status'], 2)
        self.assertEqual(students[4]['status'], 3)
        
    def test_one_student_multiple_schools_ttc(self):
        applications = {
                1: [{'uid': 4, 'score': 700}, {'uid': 3, 'score': 700}, {'uid': 2, 'score': 700}, {'uid': 1, 'score': 700}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2,3,4],
                      'VACANTE_1SEM': [1,1,1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 4)
        
    def test_cycle_students_ttc(self):
        applications = {
                1: [{'uid': 2, 'score': 700}, {'uid': 1, 'score': 800}],
                2: [{'uid': 3, 'score': 600}, {'uid': 2, 'score': 800}, {'uid': 4, 'score': 800}],
                3: [{'uid': 4, 'score': 700}, {'uid': 3, 'score': 800}],
                4: [{'uid': 3, 'score': 700}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2,3,4],
                      'VACANTE_1SEM': [1,1,1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 2)
        self.assertEqual(students[2]['status'], 3)
        self.assertEqual(students[3]['status'], 4)
        self.assertEqual(students[4]['status'], 'none')
    
    def test_da_vs_ttc(self):
        applications = {
                1: [{'uid': 2, 'score': 600}, {'uid': 1, 'score': 800}, {'uid': 3, 'score': 600}],
                2: [{'uid': 1, 'score': 600}, {'uid': 2, 'score': 800}, {'uid': 3, 'score': 700}],
                3: [{'uid': 1, 'score': 700}, {'uid': 2, 'score': 700}, {'uid': 3, 'score': 800}]}
        students = simple_setup_applications(applications)
        
        schooldata = {'CODIGO': [1,2,3],
                      'VACANTE_1SEM': [1,1,1]
        }
        schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])
        universities = prep.prepare_universities(schooldata)
        
        students, universities = ttc.execute(students, universities)
        self.assertEqual(students[1]['status'], 2)
        self.assertEqual(students[2]['status'], 1)
        self.assertEqual(students[3]['status'], 3)
        

if __name__ == '__main__':
    unittest.main()