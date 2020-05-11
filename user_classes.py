#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 18 12:12:23 2019

@author: peterawest




"""
from numpy.random import multinomial
import numpy as np
import random


def sample_user_class(user,classes):
    """ This samples a user class from the list 'classes' and 
    applies it to the given user (should be a list of posts)
    
    a user class should be a tuple of the form:
        (p, p_t, (p_o_0, f_0) , (p_o_1, f_1))
        with:
            p is the probability of being assigned to this class
            p_t is the probability of treatment in this class
            p_o_0 is probability of pos outcome given no treatment (control)
            f_0 is synthetic function applied to control individuals
            
            p_0_1 and f_1 are analogous for treatment groups
    
    """
    
    probs = [c[0] for c in classes]
    
    assert(sum(probs) == 1.)
    
    c = int(np.nonzero(multinomial(1, probs))[0] )
    # sample a class:
    (_, p_t, (p_o_0, f_0) , (p_o_1, f_1)) = classes[ c ]
    
    treat = random.random() < p_t
    
    if treat:
        p_o = p_o_1
        f = f_1
    else: 
        p_o = p_o_0
        f = f_0
    
    user = f(user)
    outcome = random.random() < p_o
    
    return user, treat, outcome, c
    
    
    