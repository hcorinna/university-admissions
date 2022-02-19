#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:28:55 2019

@author: hertweck
"""

import numpy as np
import sys
import scipy.special
from sklearn.metrics import ndcg_score, dcg_score

def get_average_score(program, year):
    return program['average_score'].get(year-1, 0)

def get_risk(score, program, year_range):
    admitted_scores = []
    for application_year in year_range:
        admitted_scores.extend(program['admitted_scores'].get(application_year, []))
    risk = 0
    if len(admitted_scores) > 0:
        risk = sum(score < admitted_score for admitted_score in admitted_scores) / len(admitted_scores)
    return risk

def get_expected_utility(average_score, risk):
    if risk != 0:
        expected_utility = average_score * risk
    else:
        expected_utility = average_score * sys.float_info.epsilon
    return expected_utility

def get_probabilities(score, program_id, clfs):
    score = np.array([score])[:,np.newaxis]
    clf = clfs.get(program_id, None)
    if clf:
        return clfs[program_id].predict_proba(score).ravel()
    return [0,1]

def get_simgoid_expected_utility(average_score, sigmoid_security):
    if sigmoid_security != 0:
        expected_utility = average_score * sigmoid_security
    else:
        expected_utility = average_score * sys.float_info.epsilon
    return expected_utility

def calculate_ranking_quality(ranked_preferences, ranking_function):
    quality = ranking_function(ranked_preferences, 1)
    risk = ranking_function(ranked_preferences, 2)
    sigmoid_risk = ranking_function(ranked_preferences, 3)
    utility = ranking_function(ranked_preferences, 4)
    sigmoid_utility = ranking_function(ranked_preferences, 5)
    return (quality, risk, sigmoid_risk, utility, sigmoid_utility)

def calculate_kendall_coefficients(ranked_preferences):
    return calculate_ranking_quality(ranked_preferences, calculate_kendall_coefficient)

def calculate_ndcg_scores(ranked_preferences):
    return calculate_ranking_quality(ranked_preferences, calculate_ndcg_score)

# =============================================================================
# def calculate_ndcg_scores_logbase(ranked_preferences, log_base=2):
#     quality = calculate_ndcg_score_logbase(ranked_preferences, 1, log_base)
#     risk = calculate_ndcg_score_logbase(ranked_preferences, 2, log_base)
#     utility = calculate_ndcg_score_logbase(ranked_preferences, 3, log_base)
#     sigmoid_risk = calculate_ndcg_score_logbase(ranked_preferences, 4, log_base)
#     sigmoid_utility = calculate_ndcg_score_logbase(ranked_preferences, 5, log_base)
#     return (quality, risk, utility, sigmoid_risk, sigmoid_utility)
# =============================================================================

def calculate_kendall_coefficient(ranked_preferences, preference_index, reverse=True):
    """
    - preference_index: 1 = order by program quality, 2 = order by risk, 3 = order by sigmoid risk
    - reverse: if set to True, then the list should be sorted from high to low
    """
    kendall_coefficients = []
    for preferences_application in ranked_preferences.values():
        ordering = [preferences[preference_index] for preferences in preferences_application]
        kendall = 0
        for i in range(len(ordering)-1):
            for j in range(i+1,len(ordering)):
                if reverse:
                    kendall += np.sign(int(ordering[i]) - int(ordering[j]))
                else:
                    kendall += np.sign(int(ordering[j]) - int(ordering[i]))
        kendall /= scipy.special.binom(len(ordering), 2)
        kendall_coefficients.append(kendall)
    return kendall_coefficients

def calculate_ndcg_score(ranked_preferences, preference_index):
    """
    - preference_index: 1 = order by program quality, 2 = order by risk, 3 = order by sigmoid risk
    """
    ndcg_scores = []
    for preferences_application in ranked_preferences.values():
        true_relevance = range(len(preferences_application), 0, -1)
        ordering = [preferences[preference_index] for preferences in preferences_application]
        ndcg = ndcg_score([true_relevance], [ordering])
        ndcg_scores.append(ndcg)
    return ndcg_scores

# =============================================================================
# def calculate_ndcg_score_logbase(ranked_preferences, preference_index, log_base=2):
#     """
#     - preference_index: 1 = order by program quality, 2 = order by risk, 3 = order by sigmoid risk
#     """
#     ndcg_scores = []
#     for preferences_application in ranked_preferences.values():
#         true_relevance = range(10, 10-len(preferences_application), -1)
#         ordering = [preferences[preference_index] for preferences in preferences_application]
#         idcg = dcg_score([true_relevance], [true_relevance], log_base=log_base)
#         dcg = dcg_score([true_relevance], [ordering], log_base=log_base)
#         ndcg = float(dcg) / float(idcg)
#         ndcg_scores.append(ndcg)
#     return ndcg_scores
# =============================================================================

def get_essential_measures_distribution(preferences_applications):
    quality = [preferences[1] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    risk = [preferences[2] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    sigmoid_risk = [preferences[3] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    return [quality, risk, sigmoid_risk]

def get_measures_distribution(preferences_applications):
    quality = [preferences[1] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    risk = [preferences[2] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    sigmoid_risk = [preferences[3] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    expected_utility = [preferences[4] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    sigmoid_expected_utility = [preferences[5] for preferences_application in preferences_applications.values() for preferences in preferences_application]
    return (quality, risk, expected_utility, sigmoid_risk, sigmoid_expected_utility)

def average_prestige_of_preferences(all_preferences):
    return [np.nanmean([preferences[1] for preferences in preferences_application]) for preferences_application in all_preferences.values()]

def calculate_measures_by_preference(preferences_applications):
    """
    - preferences_applications: {index: [(program_id, average_score, risk, expected_utility)]}
    """
    qualities = {}
    risks = {}
    expected_utilities = {}
    for preferences_application in preferences_applications.values():
        for choice in range(len(preferences_application)):
            qualities[choice] = qualities.get(choice,[])
            qualities[choice].append(preferences_application[choice][1])
            risks[choice] = risks.get(choice,[])
            risks[choice].append(preferences_application[choice][2])
            expected_utilities[choice] = expected_utilities.get(choice,[])
            expected_utilities[choice].append(preferences_application[choice][3])
    return (qualities, risks, expected_utilities)

def calculate_proportion_risky_programs(preferences_applications):
    risky_programs_proportions = []
    for preferences_application in preferences_applications.values():
        if preferences_application:
            risky_programs_proportion = sum(preferences[2] >= 0.65 for preferences in preferences_application) / len(preferences_application)
            risky_programs_proportions.append(risky_programs_proportion)
    return risky_programs_proportions

def filter_dict_by_keys(data, keys):
    return {key:value for key, value in data.items() if key in keys}

def filter_dict_by_key_lists(data, key_lists):
    for keys in key_lists:
        yield filter_dict_by_keys(data, keys)