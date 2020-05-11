#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 16:37:10 2019

@author: peterawest
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jun 23 15:21:12 2019

@author: peterawest
"""

import numpy as np
from nltk.tokenize import word_tokenize
from nltk import ngrams
import random
import torch



def list_find(l, target):
    if target is not None and target in l:
        return l.index(target)
    else: 
        return -1

def get_target_post(post, target = None):
    '''If target is not none, then only return features up to target and return whether or not the
    target is present'''
    
    
    
    word_tokenized = word_tokenize(post.lower())
    
    
    tokenized_string = ' '.join(word_tokenized)
    
    target_ind = list_find(tokenized_string,target)
    if target_ind == -1:
        has_target = False
        post_return = '[CLS] ' + post.lower()
    else: # if it has target, set bool and target and beyond from word_tokenized list
        has_target = True
        word_tokenized = word_tokenize(tokenized_string[:target_ind])
        post_return = '[CLS] ' + tokenized_string[:target_ind].lower()
    
    
    
    
    return post_return, has_target

def get_post_list(post_list, tokenizer, target = None, pretokenize = False):
    '''target is the target treatment feature. If not none, this just returns the features
    before the target, in a tuple where the second entry is whether or not the target is present:
    
    (features, True) or (features, False)
    
    Assumptions: posts are in chronological order
    
    So after the treatment is observed, no more posts are processed (all confounds are pre-treatment)
    '''
    
    posts = []
    for post in post_list:
        post_body, has_target = get_target_post(post,target=target)
        
        if pretokenize:
            
            with torch.no_grad():
                posts += [tokenizer.tokenize(post_body)]
        else:
            posts += [post_body]
        
        # if target is in this post, break before we include the target in grams
        if has_target:
            break
        
        
        
        

    
    return posts, has_target

def set_word(gram, feature_vector, gram2ind):
    '''For vocab2ind and feature_vector, return a new feature vector with this
    word set to true'''
    if gram in gram2ind.keys():
        feature_vector[gram2ind[gram]] = 1
    return feature_vector    








def get_features_HBERT(Users, tokenizer, pretokenize = False):
    
#    for i, user in enumerate(Users):
#        for j,_ in enumerate(user['T0']):
#            Users[i]['T0'][j]['body'] = Users[i]['T0'][j]['body'].lower()    



    features_by_user = []

    for i, user in enumerate(Users): # do it for MH
        posts, _ = get_post_list(user, tokenizer, pretokenize = pretokenize)
        
        
        features_by_user += [posts] #'features':feature_vector, 'treatment':has_target, 'outcome':False}

    X = [] 

    for i,_ in enumerate(Users):
        
        X += [features_by_user[i]]

    return X