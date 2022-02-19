#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 18 12:27:55 2020

@author: hertweck
"""

from nltk.tokenize import RegexpTokenizer
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from nltk.probability import FreqDist
from fair_bonus_policy.utils import strings

def find_commonalities_in_careers(careers, columns):
    results = {}
    for column in columns:
        if column == 'CARRERA':
            values = get_career_descriptors(careers)
        else:
            values = [program[column] for index, program in careers.iterrows()]
        results[column] = FreqDist(values)
    return results
        
def get_career_descriptors(careers):
    stemmer = SnowballStemmer('spanish')
    stop_words=set(stopwords.words("spanish"))
    stop_words.update(['licenciatura', 'licenciatur', 'licenci'])
    descriptors = []
    for index, career in careers.iterrows():
        career_descriptors = []
        career_title = career['CARRERA']
        without_accents = strings.strip_accents(career_title)
        lower_case = without_accents.lower()
        punctuation_tokenizer = RegexpTokenizer(r'\w+')
        tokenized_punctuation = punctuation_tokenizer.tokenize(lower_case)
        for w in tokenized_punctuation:
            if w not in stop_words and len(w) > 1 and not w.startswith('licenc'):
                if w in abbreviations:
                    w = abbreviations[w]
                stemmed = stemmer.stem(w)
                career_descriptors.append(stemmed)
        career_descriptors = set(career_descriptors)
        descriptors.extend(career_descriptors)
    return descriptors

abbreviations = dict({
    'lic': 'licenciado',
    'cs': 'ciencias',
    'ing': 'ingenieria',
    'inf': 'informacion',
    'sist': 'sistemas',
    'cont': 'control',
    'gest': 'gestion',
    'pedag': 'pedagogia',
    'ped': 'pedagogia',
    'ed': 'educacion',
    'educ': 'educacion',
    'vet': 'veterinaria',
    'adm': 'administracion',
    'univ': 'universitario',
    'ejec': 'ejecucion',
    'mat': 'matematica',
    'matematicas': 'matematica',
    'hist': 'historia',
    'soc': 'sociales',
    'ling': 'linguistica',
    'trad': 'traduccion',
    'bio': 'biologia',
    'biol': 'biologia',
    'fis': 'fisica',
    'qui': 'quimica',
    'dif': 'diferencial',
    'prob': 'problemas',
    'aud': 'audicion',
    'geo': 'geografia',
    'com': 'comunicacion',
    'gral': 'general',
    'ingl': 'ingles',
    'rec': 'recreacion',
    'nat': 'naturales',
    'len': 'lengua',
    'leng': 'lengua',
    'especial': 'especiales',
    'esp': 'especiales',
    'def': 'deficiencia',
    'refrig': 'refrigeracion',
    'vesp': 'vespertino'
     })
    