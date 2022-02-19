#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:51:29 2019

@author: hertweck
"""

import unittest
from pandas import DataFrame
import numpy as np
import prepare_data as prep

class TestPrepareStudents(unittest.TestCase):
    
    def setUp(self):
        applications = {'id': [1, 2, 3],
        'career_code_1': [1, 2, 1],
        'career_code_2': [np.nan, np.nan, 2],
        'career_code_3': [np.nan, np.nan, np.nan],
        'career_code_4': [np.nan, np.nan, np.nan],
        'career_code_5': [np.nan, np.nan, np.nan],
        'career_code_6': [np.nan, np.nan, np.nan],
        'career_code_7': [np.nan, np.nan, np.nan],
        'career_code_8': [np.nan, np.nan, np.nan],
        'career_code_9': [np.nan, np.nan, np.nan],
        'career_code_10': [np.nan, np.nan, np.nan],
        'weighted_average_score_1': [800, np.nan, 700],
        'weighted_average_score_2': [np.nan, np.nan, 700],
        'weighted_average_score_3': [np.nan, np.nan, np.nan],
        'weighted_average_score_4': [np.nan, np.nan, np.nan],
        'weighted_average_score_5': [np.nan, np.nan, np.nan],
        'weighted_average_score_6': [np.nan, np.nan, np.nan],
        'weighted_average_score_7': [np.nan, np.nan, np.nan],
        'weighted_average_score_8': [np.nan, np.nan, np.nan],
        'weighted_average_score_9': [np.nan, np.nan, np.nan],
        'weighted_average_score_10': [np.nan, np.nan, np.nan],
        'income_decile': [4, 8, 10]
        }

        self.applications = DataFrame(applications,columns= ['id', 
           'career_code_1', 'career_code_2',   'career_code_3',
           'career_code_4', 'career_code_5', 'career_code_6',
           'career_code_7', 'career_code_8', 'career_code_9', 'career_code_10',
           'weighted_average_score_1', 'weighted_average_score_2',
           'weighted_average_score_3', 'weighted_average_score_4',
           'weighted_average_score_5', 'weighted_average_score_6',
           'weighted_average_score_7', 'weighted_average_score_8',
           'weighted_average_score_9', 'weighted_average_score_10',
           'income_decile'
           ])
    
        schooldata = {'CODIGO': [1,2],
                      'VACANTE_1SEM': [1,1]
        }
        
        self.schooldata = DataFrame(schooldata,columns= ['CODIGO','VACANTE_1SEM'])

    def test_same_number_of_students(self):
        students = prep.prepare_students(self.applications)
        self.assertEqual(len(students), len(self.applications))
        
    def test_added_ses(self):
        students = prep.prepare_students(self.applications)
        self.assertEqual(students[1]['ses'], 'low')
        self.assertEqual(students[2]['ses'], 'high')
        self.assertEqual(students[3]['ses'], 'high')

    def test_apply_bonus(self):
        students = prep.prepare_students(self.applications)
        bonus = 10
        prep.apply_bonus(students, bonus)
        for preference in students[1]['preferences'].values():
            self.assertEqual(preference['score'], preference['score_before_bonus'] + bonus)
        for preference in students[2]['preferences'].values():
            self.assertEqual(preference['score'], preference['score_before_bonus'])
        for preference in students[3]['preferences'].values():
            self.assertEqual(preference['score'], preference['score_before_bonus'])
            
if __name__ == '__main__':
    unittest.main()