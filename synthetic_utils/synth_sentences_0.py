#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 10:00:22 2019

@author: peterawest
"""

## Death sentences 

treat_words_death = ['Mom', 'Mother', 'Mama', 'Father', 'Dad', 
                     'Papa', 'Brother', 'Wife', 'girlfriend', 
                     'partner', 'spouse', 'husband', 'son', 
                     'daughter', 'best friend'] 

control_words_death = ['lawnmower', 'computer', 'laptop', 
                       'phone', 'iphone', 'coffee machine', 
                       'washer', 'dryer', 'dishwasher', 'TV', 'car']

treat_sentences_death = ['My {} just died', 
                         'I just found out my {} died',
                         'My {} died last weekend',
                         'What do you do when your {} dies? This happened to me.' ,
                         'Has anyone else had a {} die recently?',
                         'I lost my {} yesterday.',
                         'My {} passed away recently.',
                         'I am in shock. My {} is gone.']

control_sentences_death = ['My {} just died', 
                         'I just found out my {} died',
                         'My {} died last weekend',
                         'What do you do when your {} dies? This happened to me.' ,
                         'Has anyone else had a {} die recently?',
                         'Where should I get a new {}? Mine just died.'
                         'Ugh my {} died!']


## Serious Illness sentences

treat_words_sick = ['cancer', 'leukemia', 'HIV', 'AIDS', 'Diabetes',
                    'lung cancer', 'stomach cancer', 'skin cancer',
                    'parkinson’s']

control_words_sick = ['a cold', 'the flu', 'a stomach bug', 'a head cold']


treat_sentences_sick = ['The doctor told me I have {}',
                        'I was at the hospital earlier and I have {}.',
                        'I got diagnosed with {} last week.',
                        'Have anyone here dealt with {}? I just got diagnosed.',
                        'How should I handle a {} diagnosis?',
                        'How do I tell my parents I have {}?']

control_sentences_sick = ['The doctor told me I have {}',
                        'I was at the hospital earlier and I have {}.',
                        'I came down with {}.',
                        'The doctor told me to get rest because I have {}.',
                        'Apparently I caught {}.']


## Isolation

treat_sentences_isolation = ['My friends stopped talking to me.',
                             'My wife just left me.',
                             'My parents kicked me out of the house today.',
                             'I feel so alone, my last friend said they needed to stop seeing me.',
                             'My partner decided that we shouldn’t talk anymore last night.' ,
                             'My folks just cut me off, they won’t talk to me anymore.' ,
                             'I just got a message from my brother that said he can’t talk to me anymore. He was my last contact in my family.',
                             'My last friend at work quit, now there’s no one I talk to reguraly.',
                             'I tried calling my Mom but she didn’t pick up the phone. I think my parents may be done with me.',
                             'I got home today and my partner was packing up to leave. Our apartment feels so empty now.']


from random import sample    

def sample_template_sentence(template_list, word_list):
    template = sample(template_list,1)[0]
    word = sample(word_list,1)[0]
    return template.format(word)
    
