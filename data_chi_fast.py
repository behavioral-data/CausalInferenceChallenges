#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 09:32:56 2019

@author: peterawest


This dataloader only works for synthetic data (not real targets)

It's meant to be a faster version, that also includes unigram option
"""

import numpy as np
from nltk.tokenize import word_tokenize
from nltk import ngrams
import random
import time



def list_find(l, target):
    if target is not None and target in l:
        return l.index(target)
    else: 
        return -1

def get_grams_post(post, include_bigrams = True):
    '''If target is not none, then only return features up to target and return whether or not the
    target is present'''
    
    if 'bigrams' and 'unigrams' in post:
        grams = post['unigrams']
        
        if include_bigrams:
            grams += post['bigrams']
    else:
        post = post['body']
        grams = []
        word_tokenized = word_tokenize(post.lower())
        
     
        grams += list(set(ngrams(word_tokenized,1)))
        
        if include_bigrams:
            grams += list(set(ngrams(word_tokenized,2)))
    
    return grams
    
    
    
    
 

def get_grams_post_list(post_list, include_bigrams=True):
    '''target is the target treatment feature. If not none, this just returns the features
    before the target, in a tuple where the second entry is whether or not the target is present:
    
    (features, True) or (features, False)
    
    Assumptions: posts are in chronological order
    
    So after the treatment is observed, no more posts are processed (all confounds are pre-treatment)
    '''

    
    grams = []
    for post in post_list:
        
        
        grams_post = get_grams_post(post,include_bigrams=include_bigrams)
        
        # if target is in this post, break before we include the target in grams

        
        grams += grams_post
        
        

    grams = list(set(grams))
    
    return grams

def set_word(gram, feature_vector, gram2ind, counts = False):
    '''For vocab2ind and feature_vector, return a new feature vector with this
    word set to true'''
    
    
    if gram in gram2ind.keys():
        if counts:
            feature_vector[gram2ind[gram]] = feature_vector[gram2ind[gram]] + 1
        else:
            feature_vector[gram2ind[gram]] = 1
    return feature_vector    









def get_features_chi(Users_full_posts, include_bigrams = True, counts = False, min_counts = 10):
    
#    for i, user in enumerate(Users):
#        for j,_ in enumerate(user['T0']):
#            Users[i]['T0'][j]['body'] = Users[i]['T0'][j]['body'].lower()    

    # First, get all words that at least 10 users say as our complete dicitonary
    #
    # So we go through each user and count number of instances of each feature
    vocab_raw = {}
    
    user_grams = []
    

    t1 = 0
    t2 = 0
    t3 = 0
    
    t_temp = time.time()

    for user in Users_full_posts: # do it for MH
        
        t1 += time.time() - t_temp
        t_temp = time.time()
        
        grams = get_grams_post_list(user, include_bigrams=include_bigrams)
        grams = list(set(grams)) # no repeats per user

        

        user_grams += [grams]
        
        
        t2 += time.time() - t_temp
        t_temp = time.time()
        
        for gram in grams:
            if gram in vocab_raw:
                vocab_raw[gram] = vocab_raw[gram] + 1
            else:
                vocab_raw[gram] = 1   
        t3 += time.time() - t_temp
        t_temp = time.time()
        
    print('times: t1 {}, t2 {}, t3 {}'.format(t1,t2, t3))


    vocab = [word for word in vocab_raw.keys() if vocab_raw[word] >= min_counts] ## only keep vocab with more than 10 users using it
    print('RAW VOCAB SIZE: {}'.format(len(vocab_raw)))
    print('VOCAB SIZE: {}'.format(len(vocab)))
    gram2ind = {gram:ind for ind,gram in enumerate(vocab)}


    features_by_user = []

    for i, user in enumerate(Users_full_posts): # do it for MH
#        grams, has_target = get_grams_post_list(user, include_bigrams=include_bigrams)
#        grams = list(set(grams)) # no repeats per user
        
        grams = user_grams[i]

        feature_vector = np.zeros(len(gram2ind))

        for gram in grams:
            feature_vector = set_word(gram, feature_vector, gram2ind, counts = counts)

        # outcome is whether or not they use suicide watch in T1
        features_by_user += [feature_vector] #'features':feature_vector, 'treatment':has_target, 'outcome':False}

    ## construct data matrix
    n_features = len(gram2ind)
    n_samples = len(features_by_user)
    X = np.zeros((n_samples, n_features))
    for i,_ in enumerate(Users_full_posts):
        X[i,:] = features_by_user[i]

    return X
