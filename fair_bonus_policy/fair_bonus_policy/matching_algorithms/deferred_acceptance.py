#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 12:36:02 2019

@author: hertweck
"""

def execute(students: dict, programs: dict):
    """Apply the deferred acceptance algorithm for matching. 
    Students and programs have to have a special format which is obtained 
    by calling 
    - prepare_data.prepare_students
    - prepare_data.prepare_universities"""
    __initialize_students(students)
    __initialize_programs(programs)
    rejected = __get_rejected_students(students)
    while rejected:
        for student in rejected:
            __apply_to_next_preferred_school(student, programs)
        for program_id, program in programs.items():
            __order_students_by_score(program)
            __tentatively_accept_best_students(program, students)
        rejected = __get_rejected_students(students)
    return students, programs


def __initialize_students(students: dict):
    """Set student-specific variables for DA algorithm."""
    for student in students.values():
        if len(student['preferences']) >= 1:
            student['status'] = 'rejected'
            student['next_preferred_school'] = 1
        else:
            student['status'] = 'none'
            student['next_preferred_school'] = None

def __initialize_programs(programs: dict):
    """Set program-specific variables for DA algorithm."""
    for program in programs.values():
        program['accepted'] = []
        program['applied'] = []
        
def __get_rejected_students(students: dict):
    """Return list of students who have not yet been accepted by the DA algorithm."""
    return [student for student in students.values() if student['status'] == 'rejected']

def __apply_to_next_preferred_school(student: dict, programs: dict):
    """Add student to the program's short list."""
    next_program = student['preferences'][student['next_preferred_school']]
    next_program_id = next_program['institution_id']
    programs[next_program_id]['accepted'].append({'student_id': student['student_id'], 'score_before_bonus': next_program['score_before_bonus'], 'score': next_program['score']})
    programs[next_program_id]['applied'].append({'student_id': student['student_id'], 'score_before_bonus': next_program['score_before_bonus'], 'score': next_program['score']})
    
def __order_students_by_score(program: dict):
    """Order the program's short list of students by their score."""
    program['accepted'].sort(key=lambda s: s['score'], reverse=True)

def __tentatively_accept_best_students(program: dict, students: dict):
    """Accept best students within quota and reject the others."""
    accepted_students = program['accepted'][:program['quota']]
    rejected_students = program['accepted'][program['quota']:]
    for accepted in accepted_students:
        student = students[accepted['student_id']]
        __tentatively_accept(student, program)
    for rejected in rejected_students:
        student = students[rejected['student_id']]
        __reject(student)
    program['accepted'] = accepted_students
    
def __tentatively_accept(student: dict, program: dict):
    """Set status of student to the program's ID to indicate tentative acceptance."""
    student['status'] = program['institution_id']

def __reject(student: dict):
    """Set status of student to rejected and change pointer to next preferred school if existing."""
    if len(student['preferences']) > student['next_preferred_school']:
        student['status'] = 'rejected'
        student['next_preferred_school'] += 1
    else:
        student['status'] = 'none'

def __remove_rejected_students(program: dict):
    """Remove the rejected students from the program's short list."""
    program['accepted'] = program['accepted'][:program['quota']]