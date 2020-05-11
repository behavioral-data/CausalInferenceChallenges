#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:29:12 2019

@author: peterawest
"""

import random





def get_split(X,T,Y, train_frac = 0.8, val_frac = None, shuffle_inds = None):
    '''
    This function takes data lists X, T, Y and splits them
    into train, validation, and optional test sets
    
    If only train frac is specified, just splits into train 
    and valid. If both train frac and valid frac are specified,
    splits into train/val/test
    
    Expects: X, T, and Y are all lists
    '''
    
    # get number of examples for each set
    n_train = int(len(Y)*train_frac)
    if val_frac is None:
        n_val = len(Y) - n_train
        n_test = 0
    else:
        n_val = int(len(Y)*val_frac)
        n_test = len(Y) - n_val - n_train
    assert((n_train + n_val + n_test) == len(Y))
    
    
    
    # reorder examples so splits will be random
    
    if shuffle_inds is None:
        inds = list(range(len(Y)))
        random.shuffle(inds)
    else:
        inds = shuffle_inds
    X = [X[ind] for ind in inds]
    T = [T[ind] for ind in inds]
    Y = [Y[ind] for ind in inds]
    
    
    
    
    # split into sets
    X_train = X[:n_train]
    T_train = T[:n_train]
    Y_train = Y[:n_train]
    
    X_val = X[n_train:n_train + n_val]
    T_val = T[n_train:n_train + n_val]
    Y_val = Y[n_train:n_train + n_val]
    
    if n_test == 0:
        return (X_train, T_train, Y_train), (X_val, T_val, Y_val)
    
    X_test = X[n_train + n_val : n_train + n_val + n_test]
    T_test = T[n_train+ n_val : n_train + n_val+ n_test]
    Y_test = Y[n_train+ n_val : n_train + n_val+ n_test]
    
    return (X_train, T_train, Y_train), (X_val, T_val, Y_val), (X_test, T_test, Y_test), inds