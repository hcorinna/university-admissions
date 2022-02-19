#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 12:18:01 2019

@author: hertweck
"""

def execute(students: dict, universities: dict):
    """Apply the deferred acceptance algorithm for matching. 
    Students and universities have to have a special format which is obtained 
    by calling 
    - prepare_data.read_applications
    - prepare_data.prepare_students
    - prepare_data.read_schooldata
    - prepare_data.prepare_universities"""
    open_universities = universities.copy()
    __initialize_students(students)
    __initialize_universities(universities, open_universities, students)
    while open_universities:
        __assign_students(students, universities, open_universities)
        __update_pointers(students, universities, open_universities)
    __finalize_students(students)
    return students, universities

def __initialize_students(students: dict):
    """Set student-specific variables for TTC algorithm."""
    for student_id, student in students.items():
        if len(student['preferences']) >= 1:
            student['status'] = 'unassigned'
            student['next_preferred_school'] = 1
        else:
            student['status'] = 'none'
            student['next_preferred_school'] = None
            
def __finalize_students(students: dict):
    """Set status of students who have not been assigned to any program to 'none'."""
    for student in students.values():
        if student['status'] == 'unassigned':
            student['status'] = 'none'

def __initialize_universities(universities: dict, open_universities: dict, students: dict):
    """Set university-specific variables for TTC algorithm."""
    for university in universities.values():
        university['preferences'] = []
    for student in students.values():
        for preference in student['preferences'].values():
            uid = preference['institution_id']
            universities[uid]['preferences'].append({'student_id': student['student_id'], 'score': preference['score']})
    for university in universities.values():
        university['preferences'].sort(key=lambda s: s['score'], reverse=True)
    for university_id, university in universities.items():
        university['spaces'] = university['quota']
        university['accepted'] = []
        if len(university['preferences']) >= 1 and university['spaces'] > 0:
            university['status'] = 'open'
            university['next_preferred_student'] = 0
        else:
            university['status'] = 'done'
            university['next_preferred_student'] = None
            del open_universities[university_id]

def __assign_students(students: dict, universities: dict, open_universities: dict):
    """Find cycles in the graph of universities' favorite students and students' favorite universities.
    Assign students in a cycle to their favorite universities."""
    universities_choice = __create_graph(open_universities)
    unvisited_universities = set(universities_choice.keys())
    visited_students = set()
    while unvisited_universities:
        current_student, round_students = __find_cycle(universities_choice, unvisited_universities, visited_students, students, universities)
        if current_student not in visited_students:
            # this means we found a new cycle!
            cycle_start = round_students.index(current_student)
            cycle_students = round_students[cycle_start:]
            __assign_students_in_cycle(cycle_students, students, universities, open_universities)
        # this leads to duplicated, but we don't care atm
        visited_students.update(round_students)
        
def __get_next_preferred_university_scores(student: dict, universities: dict):
    """Returns the id of the student's next preferred program. It also includes the student's scores for that program."""
    next_preferred_school = student['next_preferred_school']
    school_scores = student['preferences'][next_preferred_school]
    uid = school_scores['institution_id']
    if student['status'] == 'unassigned':
        while universities[uid]['spaces'] == 0 and next_preferred_school < len(student['preferences']):
            student['next_preferred_school'] += 1
            next_preferred_school = student['next_preferred_school']
            school_scores = student['preferences'][next_preferred_school]
            uid = school_scores['institution_id']
        if universities[uid]['spaces'] == 0 and next_preferred_school == len(student['preferences']):
            student['status'] = 'none'
    return school_scores

def __create_graph(open_universities: dict):
    """Link universities to their favorite student for this round."""
    universities_choice = {}
    for institution_id, university in open_universities.items():
        next_preferred_student = university['next_preferred_student']
        universities_choice[institution_id] = university['preferences'][next_preferred_student]['student_id']
    return universities_choice

def __find_cycle(universities_choice: dict, unvisited_universities: set, visited_students: set, students: dict, universities: dict):
    """Find cycle in graph of universities' favorite students and students' favorite universities."""
    round_students = []
    current_university = unvisited_universities.pop()
    current_student = universities_choice[current_university]
    while current_student not in round_students and current_student not in visited_students:
        round_students.append(current_student)
        current_university = __get_next_preferred_university_scores(students[current_student], universities)['institution_id']
        current_student = universities_choice[current_university]
        if current_university in unvisited_universities:
            unvisited_universities.remove(current_university)
        else:
            break
    return current_student, round_students

def __assign_students_in_cycle(cycle_students: list, students: dict, universities: dict, open_universities: dict):
    """Assign each student in newly found cycle to their favorite university of this round."""
    for student_id in cycle_students:
        student = students[student_id]
        scores = __get_next_preferred_university_scores(student, universities)
        uid = scores['institution_id']
        student['status'] = uid
        university = universities[uid]
        university['spaces'] -= 1
        university['accepted'].append({'student_id': student_id, 'score_before_bonus': scores['score_before_bonus'], 'score': scores['score']})
        if university['spaces'] == 0:
            university['status'] = 'filled'
            del open_universities[university['institution_id']]

def __update_pointers(students: dict, universities: dict, open_universities: dict):
    """Check for each university if they are still pointing to an available student. If not, update their preference."""
    for university_id, university in list(open_universities.items()):
        __set_next_preferred_student(university, students, open_universities)

def __set_next_preferred_student(university: dict, students: dict, open_universities: dict):
    """Update a university's preferred student if their current favorite has already been admitted to a program."""
    next_preferred_student = university['next_preferred_student']
    sid = university['preferences'][next_preferred_student]['student_id']
    while students[sid]['status'] != 'unassigned' and next_preferred_student < len(university['preferences']) - 1:
        university['next_preferred_student'] += 1
        next_preferred_student = university['next_preferred_student']
        sid = university['preferences'][next_preferred_student]['student_id']
    if students[sid]['status'] != 'unassigned' and next_preferred_student == len(university['preferences']) - 1:
        university['status'] = 'done'
        del open_universities[university['institution_id']]