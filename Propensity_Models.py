#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:20:41 2019

@author: peterawest
"""

import numpy as np
from sklearn.linear_model import Perceptron, SGDClassifier, LogisticRegressionCV
from sklearn.metrics import log_loss
from sklearn.calibration import CalibratedClassifierCV

from sklearn.model_selection import PredefinedSplit


# assuming binary treatment
def CE(P, Z):
    return -np.sum(Z*np.log(P))/len(P)

# assuming binary treatment
def accuracy(P, Z):
    return 1.*(np.sum((P > 0.5)*Z) + np.sum((P <= 0.5)*(1-Z))) / len(P)



def train_propensity_model(propensity_model, dataset, data_test=False):
    
    ## first do training and validation
    

    

    
    propensity_model.fit(dataset)
    out_dict = {}
    out_dict['learning_curve'] = propensity_model.learning_curve
    
    
    
    X = [data[0] for data in dataset.train_epoch(true_set = True)]
    Z = [data[1] for data in dataset.train_epoch(true_set = True)]
    Y = [data[2] for data in dataset.train_epoch(true_set = True)]
    
    # get training score
    P_train = propensity_model.score(X)
    out_dict['train_acc'] = accuracy(P_train, np.array(Z))
    out_dict['train_CE'] = CE(P_train, np.array(Z))
    out_dict['P_train'] = P_train
    out_dict['Z_train'] = Z
    out_dict['Y_train'] = Y
    
    
    
    X = [data[0] for data in dataset.valid_epoch()]
    Z = [data[1] for data in dataset.valid_epoch()]
    Y = [data[2] for data in dataset.valid_epoch()]
    
    # get validation score
    P_val = propensity_model.score(X)
    out_dict['val_acc'] = accuracy(P_val, np.array(Z))
    out_dict['val_CE'] = CE(P_val, np.array(Z))
    out_dict['P_val'] = P_val
    out_dict['Z_val'] = Z
    out_dict['Y_val'] = Y
    
    ## if we have test data, also return test scores
    if data_test:
        X = [data[0] for data in dataset.test_epoch()]
        Z = [data[1] for data in dataset.test_epoch()]
        Y = [data[2] for data in dataset.test_epoch()]
        
        P_test = propensity_model.score(X)
        
        out_dict['test_acc'] = accuracy(P_test, np.array(Z))
        out_dict['test_CE'] = CE(P_test, np.array(Z))
        out_dict['P_test'] = P_test
        out_dict['Z_test'] = Z
        out_dict['Y_test'] = Y
        

        
    return propensity_model, out_dict

def train_PW_model(propensity_model, dataset, data_test=False):
    
    ## first do training and validation
    

    
    propensity_model.fit(dataset)
    out_dict = {}
    out_dict['learning_curve'] = propensity_model.learning_curve
    
    
    
    X = [data[0] for data in dataset.train_epoch(true_set = True)]
    C = [data[1] for data in dataset.train_epoch(true_set = True)]
    Y = [data[2] for data in dataset.train_epoch(true_set = True)]
    
    # get training score
    P_train = propensity_model.score(X)
    out_dict['train_acc'] = accuracy(P_train, np.array(C))
    out_dict['train_CE'] = CE(P_train, np.array(C))
    out_dict['P_train'] = P_train
    
    
    X_true = [(x[0], x[2]) for x in X]
    out_dict['P_train_true'] = propensity_model.score(X_true)
#    out_dict['Z_PW_train'] = Z_PW
    #out_dict['Y_train'] = Y
    
    
    
    X = [data[0] for data in dataset.valid_epoch()]
    C = [data[1] for data in dataset.valid_epoch()]
    Y = [data[2] for data in dataset.valid_epoch()]
    
    # get training score
    P_val = propensity_model.score(X)
    out_dict['val_acc'] = accuracy(P_val, np.array(C))
    out_dict['val_CE'] = CE(P_val, np.array(C))
    out_dict['P_val'] = P_val
#    out_dict['Z_PW_val'] = Z_PW
    X_true = [(x[0], x[2]) for x in X]
    out_dict['P_val_true'] = propensity_model.score(X_true)
    
    ## if we have test data, also return test scores
    if data_test:
        X = [data[0] for data in dataset.test_epoch()]
        C = [data[1] for data in dataset.test_epoch()]
        Y = [data[2] for data in dataset.test_epoch()]
        
        # get training score
        P_test = propensity_model.score(X)
        out_dict['test_acc'] = accuracy(P_test, np.array(C))
        out_dict['test_CE'] = CE(P_test, np.array(C))
        out_dict['P_test'] = P_test
#        out_dict['Z_PW_test'] = Z_PW
        X_true = [(x[0], x[2]) for x in X]
        out_dict['P_test_true'] = propensity_model.score(X_true)
        

        
    return propensity_model, out_dict


class Logreg_Propensity_Model:
    def __init__(self, n_train, n_val, penalty='l2', scoring='neg_log_loss', class_weight='balanced'):
        
        self.n_train = n_train
        self.n_val = n_val
        
        # define validation split for sklearn
        q = np.zeros(n_train + n_val)
        q[n_train:] = -1
        
        ps = PredefinedSplit(q)
        
        self.model =  LogisticRegressionCV(Cs=20,
                        cv = ps, 
                       random_state=0,
                       penalty = penalty, 
                       scoring = scoring, 
                       class_weight=class_weight,
                      refit = False)
        
        
        
        pass

    def fit(self, X_train, Z_train, X_val, Z_val):
        
        # define full data to be used
        # THIS SHOULD MATCH n_train, n_val in definition
        X = X_train + X_val
        Z = Z_train + Z_val
        X = np.array(X)
        Z = np.array(Z)
        
        # these must match for validation to work
        assert(len(X_train) == self.n_train)
        assert(len(X_val) == self.n_val)

        self.model.fit(X,Z)
        
        
    def score(self, X):
        X = np.array(X)
        
        return self.model.predict_proba(X)[:,1]

