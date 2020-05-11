#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:25:08 2019

@author: peterawest
"""

import random
from nltk.tokenize import word_tokenize
from user_classes import sample_user_class



#def get_split(X,T,Y, train_frac = 0.8, val_frac = None):
#    '''
#    This function takes data lists X, T, Y and splits them
#    into train, validation, and optional test sets
#    
#    If only train frac is specified, just splits into train 
#    and valid. If both train frac and valid frac are specified,
#    splits into train/val/test
#    
#    Expects: X, T, and Y are all lists
#    '''
#    
#    # get number of examples for each set
#    n_train = int(len(Y)*train_frac)
#    if val_frac is None:
#        n_val = len(Y) - n_train
#        n_test = 0
#    else:
#        n_val = int(len(Y)*val_frac)
#        n_test = len(Y) - n_val - n_train
#    assert((n_train + n_val + n_test) == len(Y))
#    
#    
#    
#    # reorder examples so splits will be random
#    inds = list(range(len(Y)))
#    random.shuffle(inds)
#    X = [X[ind] for ind in inds]
#    T = [T[ind] for ind in inds]
#    Y = [Y[ind] for ind in inds]
#    
#    
#    
#    
#    # split into sets
#    X_train = X[:n_train]
#    T_train = T[:n_train]
#    Y_train = Y[:n_train]
#    
#    X_val = X[n_train:n_train + n_val]
#    T_val = T[n_train:n_train + n_val]
#    Y_val = Y[n_train:n_train + n_val]
#    
#    if n_test == 0:
#        return (X_train, T_train, Y_train), (X_val, T_val, Y_val)
#    
#    X_test = X[n_train + n_val : n_train + n_val + n_test]
#    T_test = T[n_train+ n_val : n_train + n_val+ n_test]
#    Y_test = Y[n_train+ n_val : n_train + n_val+ n_test]
#    
#    return (X_train, T_train, Y_train), (X_val, T_val, Y_val), (X_test, T_test, Y_test)


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






########
#
# Functions for CHI model features
#
#
    



#
#
#
#
#########
  
    
########
#
# Functions for synthetic data
#
    
def synthetic_data(post_dicts, post_fun_treat = None, post_fun_control = None, treatment = 0.5):
    '''
    treatment can be a fraction or a vector of binary treatment values
    post_fun_treat and  post_fun_control are 
    
    '''
    
    if post_fun_treat is None and post_fun_control is None:
        print('WARNING: no synthetic treatment given')
        
    # if given treatment frac, generate a treatment vec based on it
    if type(treatment) is float:
        assert(treatment <= 1.0 and treatment >= 0.0) # should be a valid fraction
        treat_frac = treatment
        treatment = [random.random() < treat_frac  for i in range(len(post_dicts))]
    
    for i, post_dict in enumerate(post_dicts):
        
        if treatment[i] and post_fun_treat is not None:
            post_dicts[i] = post_fun_treat(post_dict)
            
        elif not treatment[i] and post_fun_control is not None:
            post_dicts[i] = post_fun_control(post_dict)
            
    return post_dicts, treatment
    
#
#
#
########
    


def list_find(l, target):
    if target is not None and target in l:
        return l.index(target)
    else: 
        return -1

def get_target_post(post, target = None):
    '''If target is not none, then only return features up to target and return whether or not the
    target is present'''

    word_tokenized = word_tokenize(post['body'].lower())

    tokenized_string = ' '.join(word_tokenized)
    
    target_ind = list_find(tokenized_string,target)
    if target_ind == -1:
        has_target = False
        post_return = post['body'].lower()
    else: # if it has target, set bool and target and beyond from word_tokenized list
        has_target = True
        word_tokenized = word_tokenize(tokenized_string[:target_ind])
        post_return = tokenized_string[:target_ind].lower()

    return post_return, has_target

def get_target_post_list(post_list, target = None):
    '''
    Also returns the full post list in case there is precalculation in there
    
    posts: is a list of strings
    post_list: is a list of post dictionaries
    '''  
    posts = []
    
    for post in post_list:
        post_body, has_target = get_target_post(post,target=target)
        posts += [post_body]   
        # if target is in this post, break before we include the target in grams
        if has_target:
            break
    return posts, post_list, has_target






def order_users(users):
    users_out = []
    for key in users.keys():
        users_out += [users[key]]
    return users_out


def process_users_old(users_pos, users_neg, fun_treatment = None, fun_synth_treat = None, fun_synth_control = None, treated = None):
    
    Users = []
    T = []
    Y = []
    

    
    
    # if using a real treatment on the data
    if fun_treatment is not None:
        # should not specity synthetic values
        assert(fun_synth_treat is None and fun_synth_control is None and treated is None )
        
        
        for user in users_pos:
#            posts, has_target = get_target_post_list(user['T0'], target = fun_treatment)

            posts, has_target = get_target_post_list(user, target = fun_treatment)
            Users += [posts]
            T += [has_target]
            
            Y += [True]
            
        for user in users_neg:
#            posts, has_target = get_target_post_list(user['T0'], target = fun_treatment)

            posts, has_target = get_target_post_list(user, target = fun_treatment)
            Users += [posts]
            T += [has_target]
            
            Y += [False]
            
    else: # synthetic data case
        
        # if given a fraction of treated individuals
        if type(treated) is float:
            assert(treated <=1. and treated >=0.)
            T = [random.random() < treated for _ in  range(len(users_pos) + len(users_neg))]
        else: 
            T = treated
            
        count = 0
        for user in users_pos:
            
            if T[count] and fun_synth_treat:
                posts_in = fun_synth_treat(user['T0'])
            elif not T[count] and fun_synth_control:
                posts_in = fun_synth_control(user['T0'])
            else:
                posts_in = user['T0']
            

            posts, _ = get_target_post_list(posts_in, target = fun_treatment)

            Users += [posts]            

            Y += [True]
            
            count += 1
            
        for user in users_neg:
            
            if T[count] and fun_synth_treat:
                posts_in = fun_synth_treat(user['T0'])
            elif not T[count] and fun_synth_control:
                posts_in = fun_synth_control(user['T0'])
            else:
                posts_in = user['T0']
            
            posts, _ = get_target_post_list(posts_in, target = fun_treatment)
            
            Users += [posts]
  
            Y += [False]

            count += 1
        
    
    # finally, shuffle users
    inds = list(range(len(Users)))
    random.shuffle(inds)
    Users_ = [Users[i] for i in inds]
    T_ = [T[i] for i in inds]
    Y_ = [Y[i] for i in inds]
    
    return Users_, T_, Y_

def process_users_synth(users, classes, keep_class = False):
    
    Users = []
    Users_full_posts = []
    T = []
    Y = []
    
    user_class = []

    for user in users:
#        user, t, y, _ = sample_user_class(user['T0'],classes)

        
        user, t, y, c = sample_user_class(user,classes)
        user, user_full_posts, _ = get_target_post_list(user, target = None)
#        user, t, y, _ = sample_user_class(user_posts,classes)
        Users += [user]
        Users_full_posts += [user_full_posts]
        T += [t]
        Y += [y]
        
        user_class += [c]
            

    # finally, shuffle users
    inds = list(range(len(Users)))
    random.shuffle(inds)
    
    Users_ = [Users[i] for i in inds]
    Users_full_posts_ = [Users_full_posts[i] for i in inds]
    T_ = [T[i] for i in inds]
    Y_ = [Y[i] for i in inds]
    user_class_ = [user_class[i] for i in inds]

    # keep class, we return each user's class
    if keep_class:
        return Users_,Users_full_posts_, T_, Y_, user_class_
        
    return Users_,Users_full_posts_, T_, Y_



# this will allow stochastic or non-stochastic iteration over 
class Dataset():
    
    def __init__(self, X, Z, Y, PW = False, train_frac = 0.8, val_frac = 0.1):
    ## we should actually be doing the split here, just take X, Z, Y here
        
        self.PW = PW
        self.n_ex = len(X)
        
        self.X = X
        self.Z = Z
        self.Y = Y
        
        
        if self.PW:
            ## construct deterministic validation and test sets (see the arbour code)
            self.split(self.n_ex, train_frac, val_frac, PW = True)
            
            
            
            pass
        else:
            # split examples into train, val, test
            self.split(self.n_ex, train_frac, val_frac)
    
    
    
    ## this splits into train, test, valid (without permutation weighting)
    def split(self, n_ex, train_frac, val_frac=None, shuffle_inds = False, PW = False):
        # get number of examples for each set
        n_train = int(n_ex*train_frac)
        if val_frac is None:
            n_val = n_ex - n_train
            n_test = 0
        else:
            n_val = int(n_ex*val_frac)
            n_test = n_ex - n_val - n_train
        assert((n_train + n_val + n_test) == n_ex)
        
        
        
        # reorder examples so splits will be random
        
        if shuffle_inds:
            inds = list(range(n_ex))
            random.shuffle(inds)
        else:
            inds = list(range(n_ex))
            
            
            
        self.inds_train = inds[:n_train]
        self.inds_val = inds[n_train:n_train + n_val]
        
        ## if using permutation weighting, then for each validation ind,
        ## 50/50 T_ind is true or random
        if PW:
            self.C_val = [ 1*(random.random()) <0.5 for _ in self.inds_val] # pick a class for each point (PW dataset)
            self.inds_val_PW = [ (ind if (self.C_val[i] == 0) else random.sample(self.inds_val,1)[0]) for i, ind in enumerate(self.inds_val)]
        
        
        if n_test > 0:
            self.inds_test = inds[n_train + n_val : n_train + n_val + n_test]
            ## if using permutation weighting, then for each validation ind,
            ## 50/50 T_ind is true or random
            if PW:
                self.C_test = [ 1*(random.random()) <0.5 for _ in self.inds_test] # pick a class for each point (PW dataset)
                self.inds_test_PW = [ (ind if (self.C_test[i] == 0) else random.sample(self.inds_test,1)[0]) for i, ind in enumerate(self.inds_test)]

            
        
    
    def update_X(self, X_new): # updates the features of X (keeps same PW sets if applicable)
        self.X = X_new
        
    def full_dataset(self):
        for ind in range(len(self.X)):
            yield self.X[ind], self.Z[ind], self.Y[ind]

        
        
        
        
    def train_epoch(self, true_set = False, size = 1000, include_ind = False): # yields iterator with size stochastic examples (if PW, as a tuple)
        
        if self.PW:
            for _ in range(size):
                # first pick whether it's a real or fake example
                
                # whether or not it's taken from the cross product
                
                ind = self.inds_train[  random.randint(0, len(self.inds_train) - 1)] 
                
                
                permuted = random.random() <0.5
                
                ind_X = ind
                ind_Z = ind
                ind_Z_PW = ind if (permuted == 0) else random.sample(self.inds_train,1)[0]
                ind_Y = ind
                
                assert((ind_Z_PW ==  ind_Z) or permuted)
                
                if include_ind:
                    yield (self.X[ind_X], self.Z[ind_Z_PW], self.Z[ind_Z]), permuted, self.Y[ind_Y], ind_X
                else:               
                    yield (self.X[ind_X], self.Z[ind_Z_PW], self.Z[ind_Z]), permuted, self.Y[ind_Y] 
        else: 
            if true_set:
                for i in range(len(self.inds_train)):
                    
                    # get a random example from the train set
                    ind =  self.inds_train[i] 
                    if include_ind:
                        yield self.X[ind], self.Z[ind], self.Y[ind], ind
                    else:
                        yield self.X[ind], self.Z[ind], self.Y[ind]
            else:
                
                for _ in range(size):
                    
                    # get a random example from the train set
                    ind =  self.inds_train[  random.randint(0, len(self.inds_train) - 1)] 

                    if include_ind:                    
                        yield self.X[ind], self.Z[ind], self.Y[ind], ind
                    else:
                        yield self.X[ind], self.Z[ind], self.Y[ind]
        
        
    def valid_epoch(self): # gets a validation epoch, using validation data
        if self.PW:
            for i in range(len(self.inds_val)):
                # first pick whether it's a real or fake example
                
                # NOTE: should return: X, Z_PW, Z_true, and Y (Z_true so we ) 
                
                ind_X = self.inds_val[i]
                ind_Z_PW = self.inds_val_PW[i]
                ind_Z = self.inds_val[i]
                ind_Y = self.inds_val[i]
                
                assert((ind_Z_PW ==  ind_Z) or self.C_val[i])
                
                yield (self.X[ind_X], self.Z[ind_Z_PW], self.Z[ind_Z]), self.C_val[i], self.Y[ind_Y]
        else: 
            for i in range(len(self.inds_val)):
                
                # get a random example from the train set
                ind =  self.inds_val[i] 
                
                yield self.X[ind], self.Z[ind], self.Y[ind]
                
        
        
    def test_epoch(self): # same idea as validation epoch
        
        if self.PW:
            for i in range(len(self.inds_test)):
                
                # first pick whether it's a real or fake example
                ind_X = self.inds_test[i]
                ind_Z_PW = self.inds_test_PW[i]
                ind_Z = self.inds_test[i]
                ind_Y = self.inds_test[i]                
                
                
                yield (self.X[ind_X], self.Z[ind_Z_PW], self.Z[ind_Z]), self.C_test[i], self.Y[ind_Y]
        else: 
            for i in range(len(self.inds_test)):
                
                # get a random example from the train set
                ind =  self.inds_test[i] 
                
                yield self.X[ind], self.Z[ind], self.Y[ind]