#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  5 15:46:57 2020

@author: hertweck
"""

import pickle

def get_pool(pool_path):
    try:
        with open(pool_path, 'rb') as f:
            pool = pickle.load(f)
    except (FileNotFoundError, EOFError):
        pool = None
    return pool

def get_data(fp):
    try:
        with open(fp, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        data = None
    return data

def get_dict(fp):
    try:
        with open(fp, 'rb') as f:
            data = pickle.load(f)
    except (FileNotFoundError, EOFError):
        data = {}
    return data