#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 15:41:32 2019

@author: peterawest
"""

## definition of traumatic event sampling
from synthetic_utils.synth_sentences_0 import (treat_sentences_death, treat_words_death,
                                               control_sentences_death, control_words_death, 
                                               treat_sentences_sick, treat_words_sick,
                                               control_sentences_sick, control_words_sick,
                                               treat_sentences_isolation,
                                               sample_template_sentence)
from random import sample
import random 



## definition of samplers for templates/sentences of interest




#### functions to sample sentences
##
#

# append sampled sentence to the end
isolation_function_end = lambda x: x + [{'body': sample(treat_sentences_isolation,1)[0] }]
sick_function_end = lambda x: x + [{'body': sample_template_sentence(treat_sentences_sick, treat_words_sick) }]
death_function_end = lambda x: x + [{'body': sample_template_sentence(treat_sentences_death, treat_words_death) }]


# sample sentence
sample_isolation = lambda x: sample(treat_sentences_isolation,1)[0]
sample_sick = lambda x: sample_template_sentence(treat_sentences_sick, treat_words_sick)
sample_death = lambda x: sample_template_sentence(treat_sentences_death, treat_words_death)

# randomely select between 3 kinds of bad news
def bad_news_function_end(x):
    r = random.random()
    
    if r < 0.333:
        return isolation_function_end(x)
    elif r < 0.666:
        return sick_function_end(x)
    else:
        return death_function_end(x)
    
# randomely select between 3 kinds of bad news
def sample_bad_news():
    r = random.random()
    
    if r < 0.333:
        return sample_isolation(None)
    elif r < 0.666:
        return sample_sick(None)
    else:
        return sample_death(None)


identity_function = lambda x: x







#### Experiments
##
#

# Experiment 1
'''
Goal: Recognize bad events

Only 2 classes:
    1) users who just recieved bad news of illness
    2) control users (just have original reddit posts) '''

sick_class = (0.5, 0.9, (0.5,sick_function_end), (0.5,sick_function_end))

control_class = (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))

experiment_1 = [sick_class, control_class]


# Experiment 2
''' 
Goal: Discern between different rates of bad events

2 classes:
    1) get two pieces of bad news
    2) only get one piece of bad news'''
    
# add two pieces of bad news to the end
double_bad = lambda x:  bad_news_function_end(bad_news_function_end(x))
    
bad_news_class = (0.5, 0.9, (0.5,double_bad), (0.5,double_bad))
control_class =  (0.5, 0.1, (0.5,bad_news_function_end), (0.5,bad_news_function_end))
   
experiment_2 = [bad_news_class, control_class]



# Experiment 3
''' 
Goal: discern between frequent and rare mentions of 1 event

2 classes:
    1) negative thing once
    2) same negative thing more than once (let's say 3 - 5 times) '''
    

# same bad news repeated 3-5 times
treat_fun_e3 = lambda x:  x + ([{'body': sample_bad_news() }] * random.randint(3,5))

# bad new repeated once
control_fun_e3 = lambda x:  x + [{'body': sample_bad_news() }]
    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e3), (0.5,treat_fun_e3))
control_class =  (0.5, 0.1, (0.5,control_fun_e3), (0.5,control_fun_e3))
   
experiment_3 = [bad_news_class, control_class]


# Experiment 4
''' 
Goal: discern between recent and distant events

2 classes:
    1) negative thing recently
    2) negative thing at the beginning of these last few years'''
    

# same bad news repeated 3-5 times
treat_fun_e4 = lambda x:  x + [{'body': sample_bad_news() }]

# bad new repeated once
control_fun_e4 = lambda x:  [{'body': sample_bad_news() }] + x 
    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e3), (0.5,treat_fun_e3))
control_class =  (0.5, 0.1, (0.5,control_fun_e3), (0.5,control_fun_e3))
   
experiment_4 = [bad_news_class, control_class]


# Experiment 5
''' 
Goal: discern between sentences of similar structure

2 classes:
    1) negative event
    2) similarly structured benign event
    '''