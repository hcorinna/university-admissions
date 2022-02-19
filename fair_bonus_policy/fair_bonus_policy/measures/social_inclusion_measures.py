#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 10:21:55 2019

@author: hertweck
"""

import numpy as np
import pickle
from fair_bonus_policy.config import additional_data_dir

medians = pickle.load(open(str(additional_data_dir) + '/medians', 'rb'))
q1 = 548
q2 = 616
q3 = 682
nem_bounds = [q1,q2,q3]

def gender_populations(applications):
    female_population = sum(applications['gender'] == 2)
    male_population = len(applications) - female_population
    return np.array([female_population, male_population])

def income_populations(applications, year):
    income_conditions = get_conditions([medians[year]], applications['income_decile'] / applications['number_family_members'])
    return np.array([sum(income_condition) for income_condition in income_conditions])

# =============================================================================
# id_low = 2
# id_middle = 7
# ses_bounds = [id_low, id_middle]
# 
# def ses_populations(applications):
#     ses_conditions = get_conditions(ses_bounds, applications['income_decile'])
#     return np.array([sum(ses_condition) for ses_condition in ses_conditions])
# =============================================================================

def grade_populations(applications):
    nem_conditions = get_conditions(nem_bounds, applications['nem'])
    return np.array([sum(nem_condition) for nem_condition in nem_conditions])

def get_conditions(bounds, column):
    conditions = []
    conditions.append(column <= bounds[0])
    for i in range(1, len(bounds)):
        conditions.append((column > bounds[i-1]) & (column <= bounds[i]))
    conditions.append(column > bounds[len(bounds)-1])
    return conditions

def income_grade_populations(applications, year):
    income_conditions = get_conditions([medians[year]], applications['income_decile'] / applications['number_family_members'])
    nem_conditions = get_conditions(nem_bounds, applications['nem'])
    return np.array([[sum(income_condition & nem_condition) for income_condition in income_conditions] for nem_condition in nem_conditions])

def income_grade_percentile_populations(applications, nem_percentile_bounds, year):
    income_conditions = get_conditions([medians[year]], applications['income_decile'] / applications['number_family_members'])
    nem_conditions = get_conditions(nem_percentile_bounds, applications['nem'])
    nem_condition = nem_conditions[1]
    return np.array([sum(income_condition & nem_condition) for income_condition in income_conditions])

def subgroup_measures(subgroups_in_grouping, grouping, subgroups_in_population, total_population):
    proportions_in_grouping = []
    proportions_of_subgroups = []
    percentages_representation = []
    for subgroup, subgroup_in_population in zip(subgroups_in_grouping, subgroups_in_population):
        proportion_in_grouping = subgroup / len(grouping)
        proportions_of_subgroup = subgroup / subgroup_in_population
        proportion_in_applicants_pool = subgroup_in_population / total_population
        points_representation = proportion_in_grouping - proportion_in_applicants_pool
        percentage_representation = points_representation / proportion_in_applicants_pool * 100
        proportions_in_grouping.append(proportion_in_grouping)
        proportions_of_subgroups.append(proportions_of_subgroup)
        percentages_representation.append(percentage_representation)
    return proportions_in_grouping, proportions_of_subgroups, percentages_representation

def gender_measures(gender_population, total_population, admitted):
    females = sum(admitted['gender'] == 2)
    males = len(admitted) - females
    subgroups_in_grouping = [females, males]
    return subgroup_measures(subgroups_in_grouping, admitted, gender_population, total_population)

def income_measures(income_population, total_population, admitted, year):
    subgroups_in_grouping = income_populations(admitted, year)
    return subgroup_measures(subgroups_in_grouping, admitted, income_population, total_population)

def income_grade_percentile_measures(applications, admitted, nem_percentile_bounds):
    income_grades_percentile_population = income_grade_percentile_populations(applications, nem_percentile_bounds)
    subgroups_in_grouping = income_grade_percentile_populations(admitted, nem_percentile_bounds)
    return subgroup_measures(subgroups_in_grouping, admitted, income_grades_percentile_population, len(applications))
    
def income_controlled_for_grades_measures(income_grade_proportions_in_applicant_pool, total_population, admitted):
    income_grade_subgroups = income_grade_populations(admitted)
    grade_measures = []
    for income_subgroups, income_proportions_in_applicant_pool in zip(income_grade_subgroups, income_grade_proportions_in_applicant_pool):
        grade_measures.append(subgroup_measures(income_subgroups, admitted, income_proportions_in_applicant_pool, total_population))
    return grade_measures

def subgroup_base_rates(subgroups_in_admitted, subgroups_in_applied, sensitive_attribute):
    sensitive_selection_rate = subgroups_in_admitted[sensitive_attribute] / subgroups_in_applied[sensitive_attribute]
    complement_selection_rate = (np.sum(subgroups_in_admitted) - subgroups_in_admitted[sensitive_attribute]) / (np.sum(subgroups_in_applied) - subgroups_in_applied[sensitive_attribute])
    return sensitive_selection_rate, complement_selection_rate

def statistical_parity_difference(sensitive_selection_rate, complement_selection_rate):
    """
    .. math::
        P(Y = 1 | A = a) - P(Y = 1 | A \neq a)
    """
    return sensitive_selection_rate - complement_selection_rate

def disparate_impact(sensitive_selection_rate, complement_selection_rate):
    """
    .. math::
        \frac{P(Y = 1 | A = a)}{P(Y = 1 | A \neq a)}
    """
    if complement_selection_rate == 0:
        return float("inf")
    return sensitive_selection_rate / complement_selection_rate

def statistical_disparity_measures(subgroups_in_admitted, subgroups_in_applied, sensitive_attribute, unprivileged=None):
    """
    unprivileged:
        - 0 if sensitive attribute should be unprivileged
        - 1 if the rest should be considered unprivileged
        - None if subgroup with smaller acceptance rate should be considered unprivileged
    .. math::
        P(Y = 1 | A = a) - P(Y = 1 | A \neq a)\\
        \frac{P(Y = 1 | A = a)}{P(Y = 1 | A \neq a)}
    """
    sensitive_selection_rate, complement_selection_rate = subgroup_base_rates(subgroups_in_admitted, subgroups_in_applied, sensitive_attribute)
    if unprivileged == 1 or unprivileged is None and complement_selection_rate < sensitive_selection_rate:
        t = sensitive_selection_rate
        sensitive_selection_rate = complement_selection_rate
        complement_selection_rate = t
    spd = statistical_parity_difference(sensitive_selection_rate, complement_selection_rate)
    di = disparate_impact(sensitive_selection_rate, complement_selection_rate)
    return (spd, di)

def gender_statistical_disparity_measures(admitted, applied, unprivileged=None):
    subgroups_in_admitted = gender_populations(admitted)
    subgroups_in_applied = gender_populations(applied)
    return statistical_disparity_measures(subgroups_in_admitted, subgroups_in_applied, 0, unprivileged)

def income_statistical_disparity_measures(admitted, applied, year, unprivileged=None):
    subgroups_in_admitted = income_populations(admitted, year)
    subgroups_in_applied = income_populations(applied, year)
    return statistical_disparity_measures(subgroups_in_admitted, subgroups_in_applied, 0, unprivileged)

def income_controlled_for_grades_statistical_disparity_measures(admitted, applied, unprivileged=None):
    income_grade_in_admitted = income_grade_populations(admitted)
    income_grade_in_applied = income_grade_populations(applied)
    grade_measures = []
    for income_in_admitted, income_in_applied in zip(income_grade_in_admitted, income_grade_in_applied):
        grade_measures.append(statistical_disparity_measures(income_in_admitted, income_in_applied, 0, unprivileged))
    return grade_measures

def income_grade_percentile_statistical_disparity_measures(admitted, applied, nem_percentile_bounds, unprivileged=None):
    income_grades_percentile_in_admitted = income_grade_percentile_populations(admitted, nem_percentile_bounds)
    income_grades_percentile_in_applied = income_grade_percentile_populations(applied, nem_percentile_bounds)
    return statistical_disparity_measures(income_grades_percentile_in_admitted, income_grades_percentile_in_applied, 0, unprivileged)