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

sick_control_function_end = lambda x: x + [{'body': sample_template_sentence(control_sentences_sick, control_words_sick) }]

# sample sentence
sample_isolation = lambda x: sample(treat_sentences_isolation,1)[0]
sample_sick = lambda x: sample_template_sentence(treat_sentences_sick, treat_words_sick)
sample_death = lambda x: sample_template_sentence(treat_sentences_death, treat_words_death)

# randomely select between 3 kinds of bad news
#def bad_news_function_end(x):
#    r = random.random()
#    
#    if r < 0.333:
#        return isolation_function_end(x)
#    elif r < 0.666:
#        return sick_function_end(x)
#    else:
#        return death_function_end(x)
    
# randomely select between 3 kinds of bad news
def bad_news_function_end(x, probs = [0.333, 0.333]):
    r = random.random()
    
    if r < probs[0]:
        return sick_function_end(x)
        #return isolation_function_end(x)
    elif r < probs[0] + probs[1]:
        return isolation_function_end(x)        
#return sick_function_end(x)
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
    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e4), (0.5,treat_fun_e4))
control_class =  (0.5, 0.1, (0.5,control_fun_e4), (0.5,control_fun_e4))
   
experiment_4 = [bad_news_class, control_class]


# Experiment 5
''' 
Goal: discern between sentences of similar structure

2 classes:
    1) negative event (Sickness))
    2) similarly structured benign event
    '''
    
sick_class = (0.5, 0.9, (0.5,sick_function_end), (0.5,sick_function_end))
control_class =  (0.5, 0.1, (0.5,sick_control_function_end), (0.5,sick_control_function_end))
   
experiment_5 = [sick_class, control_class] 


# Experiment 6
''' 
Goal: discern between sentences of similar structure

2 classes:
    1) same negative event for everyone
    2) identity
    '''
    
bad_news_6 = sample_bad_news()
treat_fun_e6 = lambda x:  x + [{'body': bad_news_6 }]
    
sick_class = (0.5, 0.9, (0.5,treat_fun_e6), (0.5,treat_fun_e6))
control_class =  (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))
   
experiment_6 = [sick_class, control_class]  


# Experiment 7
''' 
Goal: debugging HBERT

2 classes:
    1) someone just died
    2) identity'''
    

# same bad news repeated 3-5 times
treat_fun_e7 = lambda x:   x + [{'body': sample_death(None) }]


    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e7), (0.5,treat_fun_e7))
control_class =  (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))
   
experiment_7 = [bad_news_class, control_class]


# Experiment 8
''' 
Goal: debugging HBERT

2 classes:
    1) user became isolated
    2) identity'''
    

# same bad news repeated 3-5 times
treat_fun_e8 = lambda x:   x + [{'body': sample_isolation(None) }]


    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e7), (0.5,treat_fun_e7))
control_class =  (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))
   
experiment_8 = [bad_news_class, control_class]

# Experiment 9
''' 
Goal: discern between frequent and rare mentions of 1 event

2 classes: NOTE same negative sentence for all
    1) negative thing once 
    2) same negative thing more than once (let's say 3 - 5 times) '''
    
bad_news = sample_bad_news()
# same bad news repeated 3-5 times
treat_fun_e9 = lambda x:  x + ([{'body': bad_news }] * random.randint(3,5))

# bad new repeated once
control_fun_e9 = lambda x:  x + [{'body': bad_news }]
    
bad_news_class = (0.5, 0.9, (0.5,treat_fun_e9), (0.5,treat_fun_e9))
control_class =  (0.5, 0.1, (0.5,control_fun_e9), (0.5,control_fun_e9))
   
experiment_9 = [bad_news_class, control_class]


# Experiment 10
'''
Goal: Recognize bad events

Only 2 classes:
    1) users who just recieved bad news of illness
    2) control users (just have original reddit posts) '''


sick_class = (0.3, 0.97, (0.5,sick_function_end), (0.5,sick_function_end))

death_class = (0.3, 0.7, (0.5,death_function_end), (0.5,death_function_end))

control_class = (0.4, 0.1, (0.5,identity_function), (0.5,identity_function))

experiment_10 = [sick_class, death_class, control_class]


# axes given by diversity_distractors_sequentiality_frequency_complexity
# diversity: diversity of negative sentences: 0 - 1 - 2 - 3
# distractors: are distractor sentences used? 0 - 1
# are sequential effects included? 0 - 1
# is frequency of negative post consided? 0 - 1
# how difficult is the class structure? 0 - 1 - 2
#
# standard is 1_0_0_0_0

experiment_dict = {}

# diversity 

control_class = (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))

sick_post = sample_sick(None)
treat_fun = lambda x:  x + [{'body': sick_post }]
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['0_0_0_0_0'] = [treat_class, control_class]


treat_fun = lambda x: bad_news_function_end(x, probs = [1.0,0.])
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['1_0_0_0_0'] = [treat_class, control_class]

treat_fun = lambda x: bad_news_function_end(x, probs = [0.5,0.5])
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['2_0_0_0_0'] = [treat_class, control_class]


#treat_fun = sick_function_end
treat_fun = lambda x: bad_news_function_end(x, probs = [0.33,0.33])
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['3_0_0_0_0'] = [treat_class, control_class]

control_class =  (0.5, 0.1, (0.5,sick_control_function_end), (0.5,sick_control_function_end))
experiment_dict['4_0_0_0_0'] = [treat_class, control_class]

# distractors
treat_class = (0.5, 0.9, (0.5,sick_function_end), (0.5,sick_function_end))
control_class =  (0.5, 0.1, (0.5,sick_control_function_end), (0.5,sick_control_function_end))
experiment_dict['0_1_0_0_0'] = [treat_class, control_class]


# sequential
treat_fun = lambda x:  x + [{'body': sample_sick(x) }]
control_fun = lambda x:  [{'body': sample_sick(x) }] + x 
control_class = (0.5, 0.1, (0.5,control_fun), (0.5,control_fun))
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['0_0_1_0_0'] = [treat_class, control_class]

# frequency
sick_post = sample_sick(None)
def treat_fun(x):
    for _ in range(3):
        x = x + ([{'body': sample_sick(x) }])
    return x

#treat_fun = lambda x:  x + ([{'body': sample_sick(x) }] *5)# random.randint(3,5))#[{'body': sick_post }] + [{'body': sick_post }] + [{'body': sick_post }] + [{'body': sick_post }] # ([{'body': sample_sick(x) }] * random.randint(3,5))
control_fun = lambda x:  x + [{'body': sample_sick(x) }]
control_class = (0.5, 0.1, (0.5,control_fun), (0.5,control_fun))
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))

experiment_dict['0_0_0_1_0'] = [treat_class, control_class]

sick_post = sample_sick(None)
def treat_fun(x):
    for _ in range(10):
        x = x + ([{'body': sample_sick(x) }])
    return x
treat_class = (0.5, 0.9, (0.5,treat_fun), (0.5,treat_fun))
experiment_dict['0_0_0_2_0'] = [treat_class, control_class]
# complexity

sick_class = (0.5, 0.95, (0.5,sick_function_end), (0.5,sick_function_end))
control_class = (0.5, 0.1, (0.5,identity_function), (0.5,identity_function))
experiment_dict['0_0_0_0_1'] = [sick_class, control_class]

sick_class = (0.3, 0.95, (0.5,sick_function_end), (0.5,sick_function_end))
death_class = (0.3, 0.8, (0.5,death_function_end), (0.5,death_function_end))
control_class = (0.4, 0.1, (0.5,identity_function), (0.5,identity_function))
experiment_dict['0_0_0_0_2'] = [sick_class, death_class, control_class]
