#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  4 13:02:16 2019

@author: hertweck
"""
import numpy as np
from gensim import corpora, models as gensim_models
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from fair_bonus_policy.clean_data.ClassificationPreparer import ClassificationPreparer
import sys
from abc import ABC 
  
class Predictor(ABC): 
  
    def predict_proba(self, application): 
        pass
    
    def get_model(self):
        pass
    
    def calculate_risk(self, score, program):
        admitted_scores = program['sorted_admitted_scores']
        risk = 0
        if len(admitted_scores) > 0:
            counter = number_of_elements_bigger_than(score, admitted_scores)
            risk = counter / len(admitted_scores) #TODO: Check random sample of admitted scores or sort first and binary search or precalculate proportions in dict
        return risk
    
    def calculate_expected_utility(self, average_score, risk):
        if risk != 0:
            expected_utility = average_score * risk
        else:
            expected_utility = average_score * sys.float_info.epsilon
        return expected_utility
    
def number_of_elements_bigger_than(value, array):
    start = 0
    end = len(array) - 1
    counter = end
    while start <= end:
        mid = int((start + end) / 2)
        if array[mid] <= value:
            start = mid + 1
        else:
            end = mid - 1
    counter = end+1
    return len(array) - counter

class OneTopicPerApplicationLogisticRegressionPredictor(Predictor):
    def __init__(self, preparer: ClassificationPreparer, program_types=3):
        self.program_types = program_types
        self.preparer = preparer

    def fit(self, applications_for_classifications, students):
        self.create_lda_model(students)
        self.calculate_program_probabilities_conditional_on_topic()
        self.transform_training_data(applications_for_classifications, students)
        self.logistic_regression = LogisticRegression(random_state=42, solver='lbfgs', multi_class='multinomial', max_iter=1000)
        self.logistic_regression.fit(self.X, self.y)
    
    def get_model(self):
        return self.logistic_regression
        
    def predict_proba(self, application):
        return self.logistic_regression.predict_proba(application)
    
    def predict_preferences(self, raw_application, average_number_of_applications, programs, careers):
        # should this be changed for enriched data? more columns? or do we just have repetitive columns?
        application = self.preparer.execute(raw_application)
        topic_probabilities = self.predict_proba(application).flatten()
        N = np.random.poisson(average_number_of_applications)
        if N > 10:
            N = 10
        elif N == 0:
            N = 1
        preferences = []
        for n in range(N):
            random_topic = np.random.choice(len(topic_probabilities), p=topic_probabilities)
            program_ids_probabilities = self.program_probabilities_conditional_on_topic[random_topic]
            #TODO: check that we don't draw a program we have chosen before
            program_ids = [pp[0] for pp in program_ids_probabilities]
            program_probabilities = [pp[1] for pp in program_ids_probabilities]
            if sum(program_probabilities) != 1:
                add = (1 - sum(program_probabilities)) / len(program_probabilities)
                program_probabilities = [(pp + add) for pp in program_probabilities]
            program_id = np.random.choice(program_ids, p=program_probabilities)
            #TODO: set withoutAA to False
            score = self.calculate_score(raw_application, program_id, programs, careers, True) 
            program_quality = programs[program_id]['average_score']
            program_risk = self.calculate_risk(score, programs[program_id])
            program_expected_utility = self.calculate_expected_utility(program_quality, program_risk)
            preferences.append((program_id, score, program_expected_utility))
        preferences = sorted(preferences, key=lambda x: x[2])
        preferences = [(preference[0],preference[1]) for preference in preferences]
        return preferences

    def create_lda_model(self, students):
        # order shouldn't matter, so we can just use the program IDs from the dict
        documents = [[str(preference['institution_id']) for preference in student['preferences'].values()] for student in students.values()]
        self.dictionary = corpora.Dictionary(documents)
        lda_corpus = [self.dictionary.doc2bow(document) for document in documents]
        self.lda_model = gensim_models.ldamodel.LdaModel(lda_corpus, num_topics=self.program_types, id2word = self.dictionary, passes=20)
    
    def calculate_program_probabilities_conditional_on_topic(self):
        topics = self.lda_model.get_topics()
        self.program_probabilities_conditional_on_topic = dict()
        for t in range(len(topics)):
            self.program_probabilities_conditional_on_topic[t] = []
            topic = topics[t]
            for term_id in range(len(topic)):
                program_probability = topic[term_id]
                program_id = self.dictionary[term_id]
                program_id = float(program_id.strip('\"'))
                self.program_probabilities_conditional_on_topic[t].append((program_id, program_probability))
    
    def transform_training_data(self, applications_for_classifications, students):
        topics = np.zeros(applications_for_classifications.shape[0])
        for student in students.values():
            topics_probabilities = self.lda_model.get_document_topics(self.dictionary.doc2bow([str(preference['institution_id']) for preference in student['preferences'].values()]))
            most_likely_topic = np.argmax([topic_probability[1] for topic_probability in topics_probabilities])
            topics[student['row_id']] = most_likely_topic
            
        self.X = applications_for_classifications
        self.y = topics
        
    def calculate_score(self, application, program_id, programs, careers, withoutAA=False):
        """
        If withoutAA is set to true, then we use the NEM value instead of the NEM ranking.
        Using the NEM ranking can be seen as an affirmative action policy, so excluding it makes sense to test another AA policy independently of this one.
        We are thus treading a score without the NEM ranking as putting more weight on the NEM score.
        """
        program = programs[program_id]
        career = careers.iloc[program['row_id']]
        weighted_gpa = career['%_NOTAS'] * application['nem'].values[0]
        if withoutAA:
            weighted_rank = career['%_RANK'] * application['nem'].values[0]
        else:
            weighted_rank = career['%_RANK'] * application['nem_rank'].values[0]
        weighted_language = career['%_LENG'] * application['psu_lang'].values[0]
        weighted_math = career['%_MATE'] * application['psu_mat'].values[0]
        if career['HRIA_CS_ALTERNATIVA'] == 'SI':
            weighted_social_sciences = career['%_HYCS'] * application['social_sciences_test_scores'].values[0]
            weighted_natural_sciences = career['%_CIEN'] * application['psu_cie'].values[0]
            if weighted_social_sciences >= weighted_natural_sciences:
                weighted_extra_test = weighted_social_sciences
            else:
                weighted_extra_test = weighted_natural_sciences
        else:
            if career['%_HYCS'] != 0:
                if np.isfinite(application['social_sciences_test_scores'].values[0]):
                    social_sciences_score = application['social_sciences_test_scores'].values[0]
                else:
                    social_sciences_score = self.preparer.social_sciences
                weighted_extra_test = career['%_HYCS'] * social_sciences_score
            elif career['%_CIEN'] != 0:
                if np.isfinite(application['psu_cie'].values[0]):
                    natural_sciences_score = application['psu_cie'].values[0]
                else:
                    natural_sciences_score = self.preparer.natural_sciences
                weighted_extra_test = career['%_CIEN'] * natural_sciences_score
        if np.isfinite(weighted_gpa) and np.isfinite(weighted_rank) and np.isfinite(weighted_language) and np.isfinite(weighted_math) and np.isfinite(weighted_extra_test):
            weighted_score = (weighted_gpa + weighted_rank + weighted_language + weighted_math + weighted_extra_test) / 100
            if weighted_score < career['MINIMO_PONDERADO']: #The weightedAA probably won't influence this - NEM rank pulls above average students further up, but the minimum score is usually below average.
                weighted_score = np.nan
            average_math_language = (application['psu_mat'].values[0] + application['psu_lang'].values[0]) / 2
            if average_math_language < career['MINIMO_LEN_MAT']:
                weighted_score = np.nan
        else:
            weighted_score = np.nan
        return weighted_score

class DecisionTreePredictor(Predictor):
    def __init__(self, parameters, careers, programs):
        self.criterion = parameters['criterion']
        self.max_depth = parameters['max_depth']
        self.min_samples_split = parameters['min_samples_split']
        if self.min_samples_split > 1:
            self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = parameters['min_samples_leaf']
        self.careers = careers
        self.program_ids = self.careers['CODIGO'].values
        self.programs = programs
        
    def fit(self, X_train, y_train, average=None):
        self.X_train = X_train
        self.y_train = y_train
        self.tree = DecisionTreeClassifier(criterion=self.criterion, max_depth=self.max_depth, min_samples_split=self.min_samples_split, min_samples_leaf=self.min_samples_leaf, random_state=42)
        self.tree.fit(X_train, y_train)
        self.average = average if average else np.sum(y_train) / len(X_train)
        
    def get_model(self):
        return self.tree
        
    def predict_preferences(self, sample_X, sample_applications, sample_students):
        self.X_test = sample_X
        probabilities = self.tree.predict_proba(self.X_test)
        probabilities = np.transpose(np.array(probabilities)[:,:,1])
        row_sums = probabilities.sum(axis=1)
        probabilities = probabilities / row_sums[:, np.newaxis]
        N = np.random.poisson(self.average, len(self.X_test))
        # TODO: rejection sampling?
        N[np.where(N>10)] = 10
        N[np.where(N==0)] = 1
        i = 0
        for index, application in sample_applications.iterrows():
            #n_highest = sorted(range(len(probabilities[i])), key = lambda sub: probabilities[i][sub])[-N[i]:]
            #student_predictions = self.program_ids[n_highest]
            student_predictions = np.random.choice(a=self.program_ids, size=N[i], replace=False, p=probabilities[i])
            preferences = self.add_scores_and_order_by_expected_utility(student_predictions, application)
            student = sample_students[application['id']]
            self.add_preferences_to_student(preferences, student)
            i += 1
        
    def add_scores_and_order_by_expected_utility(self, student_predictions, application):
        scores_SI = [application['nem'], application['nem_rank'], application['psu_lang'], application['psu_mat']]
        scores_NO = scores_SI.copy()
        better_science_score = max(application['social_sciences_test_scores'], application['psu_cie'])
        scores_SI.append(better_science_score)
        scores_NO.extend([application['social_sciences_test_scores'], application['psu_cie']])
        average_math_language = (application['psu_mat'] + application['psu_lang']) / 2
        preferences  = []
        for program_id in student_predictions:
            score = self.calculate_score(program_id, scores_SI, scores_NO, average_math_language)
            if np.isfinite(score):
#                ranking_score = self.programs[program_id].get('ranking_score', 0)
                program_quality = self.programs[program_id].get('average_score', 0)
                program_risk = self.calculate_risk(score, self.programs[program_id])
                program_expected_utility = self.calculate_expected_utility(program_quality, program_risk)
                preferences.append((program_id, score, program_expected_utility))
        preferences = sorted(preferences, key=lambda x: x[2], reverse=True)
        return preferences
    
    def add_preferences_to_student(self, preferences, student):
        student['preferences'] = {}
        choice = 1
        for preference in preferences:
            program_id = preference[0]
            score = preference[1]
            student['preferences'][choice] = {'institution_id': program_id, 'score_before_bonus': score, 'score': score}
            choice += 1
    
    def calculate_score(self, program_id, scores_SI, scores_NO, average_math_language):
        """
        This score calculation assumes that all nan values were imputed in a previous step.
        """
        program = self.programs[program_id]
        if average_math_language < program['minimum_language_math']:
            return np.nan
        if program['choose_between_social_and_natural_sciences'] == 'SI':
            scores = scores_SI
        else:
            scores = scores_NO
        weighted_score = np.dot(program['weights'], scores) / 100
        if weighted_score < program['minimum_score']:
            return np.nan
        return weighted_score
    
    
class RandomForestPredictor(Predictor):
    def __init__(self, parameters, careers, programs, rf=None):
        self.criterion = parameters['criterion']
        self.max_depth = parameters['max_depth']
        self.min_samples_split = parameters['min_samples_split']
        if self.min_samples_split > 1:
            self.min_samples_split = int(self.min_samples_split)
        self.min_samples_leaf = parameters['min_samples_leaf']
        self.min_samples_leaf = int(parameters['min_samples_leaf'])
        n_estimators = parameters['n_estimators']
        if n_estimators > 1:
            n_estimators = int(n_estimators)
        elif n_estimators <= 0:
            n_estimators = 1
        self.n_estimators = n_estimators
        self.careers = careers
        self.program_ids = self.careers['CODIGO'].values
        self.programs = programs
        self.rf = rf
        
    def fit(self, X_train, y_train, average=None):
        self.X_train = X_train
        self.y_train = y_train
        self.average = average if average else np.sum(y_train) / len(X_train)
        if self.rf is None:
            self.rf = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_jobs=-1, random_state=42)
            self.rf.fit(X_train, y_train)
        
    def get_model(self):
        return self.rf

    def predict_preferences(self, sample_X, sample_applications, sample_students):
        self.X_test = sample_X
        probabilities = self.rf.predict_proba(self.X_test)
        probabilities = np.transpose(np.array(probabilities)[:,:,1])
        row_sums = probabilities.sum(axis=1)
        probabilities = probabilities / row_sums[:, np.newaxis]
        N = np.random.poisson(self.average, len(self.X_test))
        N[np.where(N>10)] = 10
        N[np.where(N==0)] = 1
        i = 0
        for index, application in sample_applications.iterrows():
            #n_highest = sorted(range(len(probabilities[i])), key = lambda sub: probabilities[i][sub])[-N[i]:]
            #student_predictions = self.program_ids[n_highest]
            student_predictions = np.random.choice(a=self.program_ids, size=N[i], replace=False, p=probabilities[i])
            preferences = self.add_scores_and_order_by_probability(student_predictions, application, probabilities[i])
            student = sample_students[application['id']]
            self.add_preferences_to_student(preferences, student)
            i += 1
        
    def add_scores_and_order_by_probability(self, student_predictions, application, probabilities):
        scores_SI = [application['nem'], application['nem_rank'], application['psu_lang'], application['psu_mat']]
        scores_NO = scores_SI.copy()
        better_science_score = max(application['social_sciences_test_scores'], application['psu_cie'])
        scores_SI.append(better_science_score)
        scores_NO.extend([application['social_sciences_test_scores'], application['psu_cie']])
        average_math_language = (application['psu_mat'] + application['psu_lang']) / 2
        preferences  = []
        for program_id in student_predictions:
            score = self.calculate_score(program_id, scores_SI, scores_NO, average_math_language)
            if np.isfinite(score):
                program_id_index, = np.where(self.program_ids == program_id)
                program_probability = probabilities[program_id_index]
                preferences.append((program_id, score, program_probability))
        preferences = sorted(preferences, key=lambda x: x[2], reverse=True)
        return preferences
    
    def add_preferences_to_student(self, preferences, student):
        student['preferences'] = {}
        choice = 1
        for preference in preferences:
            program_id = preference[0]
            score = preference[1]
            student['preferences'][choice] = {'institution_id': program_id, 'score_before_bonus': score, 'score': score}
            choice += 1
    
    def calculate_score(self, program_id, scores_SI, scores_NO, average_math_language):
        """
        This score calculation assumes that all nan values were imputed in a previous step.
        """
        program = self.programs[program_id]
        if average_math_language < program['minimum_language_math']:
            return np.nan
        if program['choose_between_social_and_natural_sciences'] == 'SI':
            scores = scores_SI
        else:
            scores = scores_NO
        weighted_score = np.dot(program['weights'], scores) / 100
        if weighted_score < program['minimum_score']:
            return np.nan
        return weighted_score
    
class RandomForestRegressorPredictor(Predictor):
    def __init__(self, parameters, careers, programs, rf=None):
        n_estimators = parameters['n_estimators']
        if n_estimators > 1:
            n_estimators = int(n_estimators)
        elif n_estimators <= 1:
            n_estimators = 2
        self.n_estimators = n_estimators
        min_samples_split = parameters['min_samples_split']
        if min_samples_split > 1:
            min_samples_split = int(min_samples_split)
        elif min_samples_split <= 0:
            min_samples_split = 2
        self.min_samples_split = min_samples_split
        max_depth = parameters['max_depth']
        if max_depth is not None and max_depth <= 0:
            max_depth = 1
        self.max_depth = max_depth
        self.min_samples_leaf = int(parameters['min_samples_leaf'])
        self.careers = careers
        self.program_ids = self.careers['CODIGO'].values
        self.programs = programs
        self.rf = rf
        
    def fit(self, X_train, y_ranking_train, average=None):
        self.X_train = X_train
        self.y_ranking_train = y_ranking_train
        self.average = average if average else np.count_nonzero(y_ranking_train) / len(X_train)
        if self.rf is None:
            self.rf = RandomForestRegressor(n_estimators=self.n_estimators, min_samples_leaf=self.min_samples_leaf, max_depth=self.max_depth, min_samples_split=self.min_samples_split, n_jobs=-1, random_state=42)
            self.rf.fit(X_train, y_ranking_train)
        
    def get_model(self):
        return self.rf

    def predict_preferences(self, sample_X, sample_applications, sample_students):
        self.X_test = sample_X
        predictions = self.rf.predict(self.X_test)
        N = np.random.poisson(self.average, len(self.X_test))
        N[np.where(N>10)] = 10
        N[np.where(N==0)] = 1
        i = 0
        for index, application in sample_applications.iterrows():
            n_highest = sorted(range(len(predictions[i])), key = lambda sub: predictions[i][sub])[-N[i]:]
            student_predictions = self.program_ids[n_highest]
            preferences = self.add_scores(student_predictions, application, predictions[i])
            student = sample_students[application['id']]
            self.add_preferences_to_student(preferences, student)
            i += 1
        
    def add_scores(self, student_predictions, application, predictions):
        scores_SI = [application['nem'], application['nem_rank'], application['psu_lang'], application['psu_mat']]
        scores_NO = scores_SI.copy()
        better_science_score = max(application['social_sciences_test_scores'], application['psu_cie'])
        scores_SI.append(better_science_score)
        scores_NO.extend([application['social_sciences_test_scores'], application['psu_cie']])
        average_math_language = (application['psu_mat'] + application['psu_lang']) / 2
        preferences  = []
        for program_id in student_predictions:
            score = self.calculate_score(program_id, scores_SI, scores_NO, average_math_language)
            if np.isfinite(score):
                preferences.append((program_id, score))
        return preferences
    
    def add_preferences_to_student(self, preferences, student):
        student['preferences'] = {}
        choice = 1
        for preference in preferences:
            program_id = preference[0]
            score = preference[1]
            student['preferences'][choice] = {'institution_id': program_id, 'score_before_bonus': score, 'score': score}
            choice += 1
    
    def calculate_score(self, program_id, scores_SI, scores_NO, average_math_language):
        """
        This score calculation assumes that all nan values were imputed in a previous step.
        """
        program = self.programs[program_id]
        if average_math_language < program['minimum_language_math']:
            return np.nan
        if program['choose_between_social_and_natural_sciences'] == 'SI':
            scores = scores_SI
        else:
            scores = scores_NO
        weighted_score = np.dot(program['weights'], scores) / 100
        if weighted_score < program['minimum_score']:
            return np.nan
        return weighted_score