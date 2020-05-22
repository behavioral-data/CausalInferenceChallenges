#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun 20 17:22:37 2019

@author: peterawest
"""

import numpy as np
import matplotlib.pyplot as plt


####
#
## Code to get ATT given propensity, treatment, and outcome
#

def get_effect_ATT(P,Z,Y, K = 10, mode = 'strat'):
    if mode == 'strat':
        return get_effect_strat(P,Z,Y, K = K, value='ATT')
    elif mode == 'match':
        return get_effect_match_ATT(P,Z,Y)
    elif mode == 'iptw':
        return get_effect_iptw_ATT(P,Z,Y)
    elif mode == 'pw':
        return get_effect_pw_ATT(P,Z,Y)
    else:
        assert(False)



        
def get_effect_match(P,Z,Y, replacement=True, caliper=None, warning_tol=.1):
    '''
    full matching = each (or some?) control matched to one or more treated subjects (rosenbaum, 179) *without* replacement
    
    caliper?
    We enforce a propensity score caliper equal to 0.1 standard
    deviations of the estimated distribution, which discards any treated units for whom the
    nearest control unit is not within a suitable distance.
    from Mozer 2018: https://arxiv.org/abs/1801.00644, page 31
     -> does this mean the distribution of all (both treated and untreated) propensity scores?
     
     
    then again, rosenbaum (p. 169) recomends starting with .2
    
    vary caliber in terms of the standard deviation of the propensity scores, eg. .2 for 20% of a std away
    '''
    if not replacement:
        assert(False) # not implemented yet (but it should be)
        
    P_control = P[Z == 0]
    Y_control = Y[Z == 0]
    
    P_treat = P[Z == 1]
    Y_treat = Y[Z == 1]
    
    # if we match all the treated to the closest untreated with replacement, we have ATE on Treated (ATT)
    # if we match all the untreated to the closest treated with replacement, we have ATE on Untreated (ATU?)
    # I have no idea if this works, but let's try doing both to compute ATE
    
    diffs = [] # difference in outcome (treated - control)
    dists = [] # differenence in propensity score (for caliper application)
    
    # todo!!! also implement this without replacement
    for treat_index, p in enumerate(P_treat):        
        # get match for p (closest point in control by propensity)
        control_index = np.argmin( np.abs(P_control - p) )
        diffs.append( Y_treat[treat_index] - Y_control[control_index] )
        dists.append( np.abs( P_control[control_index] - p ) )
        
    for control_index, p in enumerate(P_control):
        # get match for p (closest point in treat by propensity)
        treat_index = np.argmin( np.abs(P_treat - p) )
        diffs.append( Y_treat[treat_index] - Y_control[control_index] )
        dists.append( np.abs( P_treat[treat_index] - p ) )
    
    
    
    
    # CALIPER SECTION IS AGNOSTIC TO REPLACEMENT IMPLEMENTATION
    # if using caliper, discard invalid matches
    if caliper is not None:        
        Valid = diffs < (caliper * P.std()) # caliper is fraction of standard deviation from the distribution
        
        #print('discarded {} by caliper'.format((1. - 1.*np.sum(Valid)/Valid.size)))
        if (1. - 1.*np.sum(Valid)/Valid.size) > warning_tol:
            print('WARNING: discarding {} of examples, exceeds tolerance of {}'.format((1. - 1.*np.sum(Valid)/Valid.size), warning_tol))
        
        diffs = np.array(diffs)[Valid]
        
    return np.average(diffs)






def get_effect_match_ATT(P,Z,Y, replacement = True, caliper = None):
    '''
    
    
    caliper = (p_tol, warning_tol)
    '''
    
    
    if not replacement:
        assert(False) # not implemented yet (but it should be)
        
    P_control = P[Z == 0]
    Y_control = Y[Z == 0]
    
    P_treat = P[Z == 1]
    Y_treat = Y[Z == 1]
    
    match_inds = []
    
    # !!! also implement this without replacement
    for p in P_treat:
        
        
        # get match for p (most similar point in control by propensity)
        match_inds += [ np.argmin(np.abs(P_control - p))  ]
        
    P_match = P_control[match_inds]

    Y_match = Y_control[match_inds]
    
    
    
    
    # CALIPER SECTION IS AGNOSTIC TO REPLACEMENT IMPLEMENTATION
    # if using caliper, discard invalid matches
    if caliper is not None:
        p_tol, warning_tol = caliper
        
        Valid = np.abs(P_treat - P_match) < p_tol
        
        print('discarded {} by caliper'.format((1. - 1.*np.sum(Valid)/Valid.size)))
        if (1. - 1.*np.sum(Valid)/Valid.size) > warning_tol:
            print('WARNING: discarding {} of examples, exceeds tolerance of {}'.format((1. - 1.*np.sum(Valid)/Valid.size), warning_tol))
        
        P_match = P_match[Valid]
        Y_match = Y_match[Valid]
        
        P_treat = P_treat[Valid]
        Y_treat = Y_treat[Valid]
        
    
    Y_diff = Y_treat - Y_match
    
    return np.average(Y_diff), (Y_diff, P_treat, P_match, Y_treat, Y_match)
        
    

def get_effect_strat_ATT(P,Z,Y, K = 10, min_max_trim = True,plot_hists = False):

    ind_sort = np.argsort(P)
    P = P[ind_sort]
    Z = Z[ind_sort]
    Y = Y[ind_sort]



    if min_max_trim:
        min_control = np.min(P[Z == 0])
        min_treat = np.min(P[Z == 1])
        
        max_control = np.max(P[Z == 0])
        max_treat = np.max(P[Z==1])

        lower_lim = max([min_control, min_treat])
        upper_lim = min([max_control, max_treat])

        ind_trim = (P <= upper_lim)*(P>= lower_lim)

        P_trim = P[ind_trim]
        Z_trim = Z[ind_trim]
        Y_trim = Y[ind_trim]
    else: # otherwise just take originals
        P_trim = P
        Z_trim = Z
        Y_trim = Y

#    print(min_control)
#    print(min_treat)
#    print(upper_lim)
#    print(max_control)
#    print(max_treat)
#    print(lower_lim)
#    plt.figure()
#    plt.hist(P_trim[Z_trim==1], label = 'treated')
#    plt.hist(P_trim[Z_trim==0],alpha = 0.3, label = 'control')
        
    n_ex = len(P_trim)

    strata_lims = []
    for j in range(K + 1):
        strata_lims += [int((1.*j/K)*n_ex)]

    strata_dicts = []
    strata_effects = []   
    strata_sizes = []
    strata_sizes_treated = []
    for i in range(K):
        
        strat_start = strata_lims[i]
        strat_end = strata_lims[i+1]

        P_strat = P_trim[strat_start:strat_end]
        Z_strat = Z_trim[strat_start:strat_end]
        Y_strat = Y_trim[strat_start:strat_end]
        

        

        control_mean = np.average(Y_strat[Z_strat==0])
        treat_mean = np.average(Y_strat[Z_strat==1])

        avg_effect_strat = treat_mean - control_mean
        n_ex_strat = strat_end - strat_start
        n_ex_strat_treated = np.sum(Z_strat==1)

        strata_effects += [avg_effect_strat]
        strata_sizes += [n_ex_strat]
        strata_sizes_treated += [n_ex_strat_treated]
        
        
        strata_dicts += [{'control_mean': control_mean,
                          'treat_mean': treat_mean,
                          'effect': avg_effect_strat, 
                          'size': n_ex_strat,
                          'start': P_trim[strat_start],
                          'end': P_trim[strat_end-1]}]
    
        if plot_hists:
            bins = list(np.linspace(P_trim[strat_start], P_trim[strat_end-1],10))
            
            plt.figure()
            plt.title('stratum {}'.format(i))
            plt.hist(P_strat[Z_strat==1], bins=bins,label = 'treated')
            plt.hist(P_strat[Z_strat==0], bins=bins, alpha = 0.3, label = 'control')
            plt.legend
        
    
    avg_effect = sum([strata_effects[i] * strata_sizes_treated[i] for i in range(len(strata_effects))])/sum(strata_sizes_treated)
    #avg_effect = sum([strata_effects[i] * strata_sizes[i] for i in range(len(strata_effects))])/sum(strata_sizes)
    return avg_effect, strata_dicts



def get_effect_strat(P,Z,Y, K = 10, min_max_trim = True,plot_hists = False, value='ATT'):

    ind_sort = np.argsort(P)
    P = P[ind_sort]
    Z = Z[ind_sort]
    Y = Y[ind_sort]



    if min_max_trim:
        min_control = np.min(P[Z == 0])
        min_treat = np.min(P[Z == 1])
        
        max_control = np.max(P[Z == 0])
        max_treat = np.max(P[Z==1])

        lower_lim = max([min_control, min_treat])
        upper_lim = min([max_control, max_treat])

        ind_trim = (P <= upper_lim)*(P>= lower_lim)

        P_trim = P[ind_trim]
        Z_trim = Z[ind_trim]
        Y_trim = Y[ind_trim]
    else: # otherwise just take originals
        P_trim = P
        Z_trim = Z
        Y_trim = Y

#    print(min_control)
#    print(min_treat)
#    print(upper_lim)
#    print(max_control)
#    print(max_treat)
#    print(lower_lim)
#    plt.figure()
#    plt.hist(P_trim[Z_trim==1], label = 'treated')
#    plt.hist(P_trim[Z_trim==0],alpha = 0.3, label = 'control')
        
    n_ex = len(P_trim)

    strata_lims = []
    for j in range(K + 1):
        strata_lims += [int((1.*j/K)*n_ex)]

    strata_dicts = []
    strata_effects = []   
    strata_sizes = []
    strata_sizes_treated = []
    for i in range(K):
        
        strat_start = strata_lims[i]
        strat_end = strata_lims[i+1]

        P_strat = P_trim[strat_start:strat_end]
        Z_strat = Z_trim[strat_start:strat_end]
        Y_strat = Y_trim[strat_start:strat_end]
        

        

        control_mean = Y_strat[Z_strat==0].mean()
        treat_mean = Y_strat[Z_strat==1].mean()

        avg_effect_strat = treat_mean - control_mean
        n_ex_strat = strat_end - strat_start
        n_ex_strat_treated = np.sum(Z_strat==1)

        strata_effects += [avg_effect_strat]
        strata_sizes += [n_ex_strat]
        strata_sizes_treated += [n_ex_strat_treated]
        
        
        strata_dicts += [{'control_mean': control_mean,
                          'treat_mean': treat_mean,
                          'effect': avg_effect_strat, 
                          'size': n_ex_strat,
                          'start': P_trim[strat_start],
                          'end': P_trim[strat_end-1]}]
    
        if plot_hists:
            bins = list(np.linspace(P_trim[strat_start], P_trim[strat_end-1],10))
            
            plt.figure()
            plt.title('stratum {}'.format(i))
            plt.hist(P_strat[Z_strat==1], bins=bins,label = 'treated')
            plt.hist(P_strat[Z_strat==0], bins=bins, alpha = 0.3, label = 'control')
            plt.legend
        
    
    if value == 'ATT':
        avg_effect = sum([strata_effects[i] * strata_sizes_treated[i] for i in range(len(strata_effects))])/sum(strata_sizes_treated)
    elif value == 'ATE':
        avg_effect = sum([strata_effects[i] * strata_sizes[i] for i in range(len(strata_effects))])/sum(strata_sizes)
    else:
        assert(False)
    #avg_effect = sum([strata_effects[i] * strata_sizes[i] for i in range(len(strata_effects))])/sum(strata_sizes)
    return avg_effect, strata_dicts


## see https://www.ncbi.nlm.nih.gov/pmc/articles/PMC3144483/ "An Introduction to Propensity Score Methods for Reducing the Effects of Confounding in Observational Studies"
def get_effect_iptw_ATT(P,Z,Y):
    
    return (1./len(P))*np.sum(Y* (  Z - (1-Z)/(1-P) ) )
    
def get_effect_iptw_ATE(P,Z,Y, normalized = True, return_weights = False):
    if normalized:
        val = np.sum(Y* (  Z/P))/np.sum(Z/P) - np.sum(Y* (  (1-Z)/(1-P)))/np.sum((1-Z)/(1-P))
        weights = (Z/P)/np.sum(Z/P) - (  (1-Z)/(1-P))/np.sum((1-Z)/(1-P))
    else:
        val = (1./ (len(P)  ) )*np.sum(Y* (  Z/P - (1-Z)/(1-P) ) )
        weights = (1./ (len(P)  ) )*(  Z/P - (1-Z)/(1-P) ) 
                                                             
    if return_weights:
        return val, weights
                                                             
    else:                                                         
        return val




def get_effect_pw_ATT(P,Z,Y, weights):
    
    treated_weights = Z == 1
    
    control_weights = (Z == 0)*(1/(1 - P))
    
    return np.sum((treated_weights)*Y)/np.sum(treated_weights) - np.sum((control_weights)*Y)/np.sum(control_weights)
    # naive
    # return (1./len(P))*np.sum(Y* (  Z - (1-Z)*weights ) )

def get_effect_pw_ATE(P,Z,Y, weights):
    
    treated_weights = (Z == 1)*(1/P)
    
    control_weights = (Z == 0)*(1/(1 - P))
    
    return np.sum((treated_weights)*Y)/np.sum(treated_weights) - np.sum((control_weights)*Y)/np.sum(control_weights)
    # naive
    # return (1./len(P))*np.sum(Y* (  Z*weights - (1-Z)*weights ) )
    
    
