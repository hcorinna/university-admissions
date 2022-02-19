#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 20:15:49 2020

@author: hertweck
"""

def program_id_format_2012(program_id):
    program_id = str(program_id)
    if len(program_id) == 4:
        program_id = program_id[:2] + '0' + program_id[2:]
    program_id = int(program_id)
    return program_id

def program_id_format_2011(program_id):
    program_id = str(program_id)
    if len(program_id) == 5 and program_id[2] == '0':
        program_id = program_id[:2] + program_id[3:]
    program_id = int(program_id)
    return program_id