#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
This script runs all models for one experiment

This corresponds to 1 value of 1 axis from the paper

E.g. this could run all models for base linguistic complexity


TO RUN:

python run_experiment_annotated.py <OPTIONS>

The base options used in the paper are:
-exp
-n_user

For a comprehensive list of the values used for this, see 
the 




TO ADD A NEW MODEL:

You must implement a feature set, model, and trainer. 

Then add an entry for your model to the model_dicts list

See existing models for example. 

"""


import argparse
from tqdm import tqdm
import time

import tempfile

import torch
import time
#from Effect_Estimate import get_effect_match_ATT, get_effect_strat_ATT
from Propensity_Models import Logreg_Propensity_Model, train_propensity_model
from data_utils import get_split, Dataset, process_users_synth
from data_RNN import Tokenizer

import os
from data_chi_fast import get_features_chi

from data_HBERT import get_features_HBERT

from pytorch_pretrained_bert import BertTokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')


from data_utils import get_split, order_users

from HBERT_simple import Hierarchical_BERT_propensity_model
from BERT_avg import Average_BERT_propensity_model

from LR_pytorch import LogReg_PT_propensity_model
from NN_pytorch import NN_PT_propensity_model

import numpy as np

from synthetic_utils.synth_sentences_0 import (treat_sentences_death, treat_words_death, 
                                               treat_sentences_sick, treat_words_sick,
                                               treat_sentences_isolation,
                                               sample_template_sentence)
from synthetic_utils.synthetic_experiments import experiment_dict

from sklearn.decomposition import LatentDirichletAllocation
from sklearn.datasets import make_multilabel_classification


def main():
    ''' Main function '''
    parser = argparse.ArgumentParser()

    ## General experimental parameters
    parser.add_argument('-exp', type=str, default = '') # which experiment to run
    parser.add_argument('-reddit_path',type=str, default='data/posts.npy') # path to reddit post data    

    parser.add_argument('-val_interval', type=int, default = 1000) # how often to evaluate models during training
    parser.add_argument('-size', type=str, default = 'med') # maximum post history length 
    parser.add_argument('-n_user', type=int, default=8000) # number of users for experiment
    parser.add_argument('-no_try', action='store_true') # whether or not to run code in a try-except
    
    

    
    
    ## arguments for training HBERT models
    parser.add_argument('-max_tokens_batch', type=int, default = 10000) # how big batches we allow BERT to run (depends on GPU)
    parser.add_argument('-lr', type=float, default = 0.00001) # learning rate for HBERT classification layers
    parser.add_argument('-bs', type=int, default = 10) # batch size for training
    parser.add_argument('-n_it', type=int, default = 8000) # number of iterations to train for
    parser.add_argument('-seq', action='store_true') 
    parser.add_argument('-temp_file_path', type=str, default='') # path to directory for temp files. if '' this is not used
    parser.add_argument('-preembed_size', type=int, default = 10) # internal hidden size
    
    
    opt = parser.parse_args()
    
    
    
    #
    #
    #
    ###################################
    ###################################
    #
    #
    #     
    """
    The first section loads the reddit user data,
    
    does some preprocessing, and carries out train/val/test split
    
    
    """
    exp_name = 'experiment_' + str(opt.exp) + '_' + opt.size
    if opt.n_user != 8000:
        exp_name += '_nuser{}'.format(opt.n_user)    

    exp_classes = experiment_dict[opt.exp]

    print(exp_classes)
    
    
    # creat data if not done already
    if (not os.path.isdir(exp_name)):

        # '/projects/bdata/datasets_peter/dataset_3/posts.npy'
        Reddit_posts = np.load(opt.reddit_path, allow_pickle=True)[0]
        

        Reddit_posts = order_users(Reddit_posts)[:opt.n_user]
        
        
        try:
            opt.size = int(opt.size)
            Reddit_posts = [user[:opt.size] for user in Reddit_posts]
            opt.size = 'size'+str(opt.size)
        except:
            if opt.size == 'xsmall':
                Reddit_posts = [user[:50] for user in Reddit_posts]
            elif opt.size == 'test':
                Reddit_posts = [user[:2] for user in Reddit_posts]
            elif opt.size == 'min':
                Reddit_posts = [user[:10] for user in Reddit_posts]
            elif opt.size == 'small':
                Reddit_posts = [user[:100] for user in Reddit_posts]
            elif opt.size == 'med':
                Reddit_posts = [user[:200] for user in Reddit_posts]
            elif opt.size == 'big':
                pass
            else:
                assert(False)
        
        print(exp_classes)
        Users, Users_full_posts, T, Y, classes = process_users_synth(Reddit_posts, #user_list, #order_users(MH2SW_posts)+ order_users(MH_posts), 
                            exp_classes, keep_class = True)
        

        
        os.mkdir(exp_name)
        
        np.save('{}/data.npy'.format(exp_name), [Users, Users_full_posts, T,Y,classes ])
        
    
    Users, Users_full_posts, T, Y, classes = tuple(np.load('{}/data.npy'.format(exp_name), allow_pickle=True))

    
    
    #
    #
    #
    ###################################
    ###################################
    #
    #
    #   
    """
    This section produces feature sets for the different models
    
    Feature sets represent some featurization of user histories
    
    e.g.
    X_chi and X_chi_counts use bag of words, with and without counts
    
    X_HBERT largely leaves user histories as text
    
    X_LDA processes X_chi_counts using LDA
    
    refer to paper for further details
    
    """
    print('starting data loading...')
    
    s_time = time.time()
    X_chi = get_features_chi(Users_full_posts)
    X_chi_counts = get_features_chi(Users_full_posts,counts = True)
    print('time = {}'.format(time.time() - s_time))
    X_chi_uni = get_features_chi(Users_full_posts, include_bigrams=False)
    X_HBERT = get_features_HBERT(Users, tokenizer, pretokenize = True)
    
    print('fitting LDA...')
    n_topics = 20
    lda = LatentDirichletAllocation(n_components=n_topics, random_state=0)
    
    X_LDA = lda.fit_transform(X_chi_counts)
    print('fit LDA')
    
    X_inds = list(range(len(Users)))
    
    dataset = Dataset(X_inds, T,Y,train_frac=0.4, val_frac=0.1)
    
    
    inds_train = [data[0] for data in dataset.train_epoch(true_set = True)]
    inds_val = [data[0] for data in dataset.valid_epoch()]
    inds_test = [data[0] for data in dataset.test_epoch()]
    
    np.save('{}/{}.npy'.format(exp_name, 'inds_dict'), {'inds_train':inds_train, 'inds_val':inds_val, 'inds_test':inds_test})


    print('{} train examples, {} val examples, {} test examples'.format(len(inds_train), len(inds_val), len(inds_test)))
    time.sleep(3)
    
    print('done data loading')
    
    
    #
    #
    #
    ###################################
    ###################################
    #
    #
    #   
    
    """
    The next section defines model_dict data structures, which are used to 
    organize training and evaluation of the models
    

    First, model dicts are defined, than added to a list of models to run, model_dicts
    """
    
    
    ## instantiate model dicts
    
    model_dict_0 = {'X':X_chi,
                   'model':LogReg_PT_propensity_model(input_dim=len(X_chi[0]), lr=0.0001, experiment_name = exp_name + '/LR_12_' + exp_name),
                   'model_name':'Logistic_Regression'}
    
    model_dict_1 = {'X':X_chi,
                   'model':NN_PT_propensity_model(input_dim=len(X_chi[0]), lr=0.0001, experiment_name = exp_name + '/NN_12_' + exp_name),
                   'model_name':'Simple_NN'}
    
    model_dict_2 = {'X':X_chi_uni,
                   'model':LogReg_PT_propensity_model(input_dim=len(X_chi_uni[0]), lr=0.0001, experiment_name = exp_name + '/LR_1_' + exp_name),
                   'model_name':'Logistic_Regression_(1gram)'}
    
    model_dict_3 = {'X':X_chi_uni,
                   'model':NN_PT_propensity_model(input_dim=len(X_chi_uni[0]), lr=0.0001, experiment_name = exp_name + '/NN_1_' + exp_name),
                   'model_name':'Simple_NN_(1gram)'}
    
    # A temporary file can be added to do some precalculation, making HBERT more efficient
    # '/projects/bdata/datasets_peter/precalc/'
    d_input = None
    if len(opt.temp_file_path ) > 0:
        d = tempfile.TemporaryDirectory(prefix=opt.temp_file_path)
        d_input = d.name + '/' + exp_name
        
    model_dict_4 = {'X':X_HBERT,
                   'model':Hierarchical_BERT_propensity_model(n_it = opt.n_it,val_interval=opt.val_interval, lr=opt.lr, batch_size=opt.bs,
                                                              h_size_sent=1000, h_size_user=1000, tokenize=False,
                                                              precalc_path= d_input,
                                                              experiment_name = exp_name + '/hbert' + exp_name,
                                                              seq = opt.seq, max_tokens_batch = opt.max_tokens_batch,
                                                              preembed_size = opt.preembed_size),
                   'model_name':'HBERT'}
    
    model_dict_5 = {'X':X_chi_counts,
                   'model':LogReg_PT_propensity_model(input_dim=len(X_chi[0]), lr=0.0001, experiment_name = 'LR_12_' + exp_name),
                   'model_name':'Logistic_Regression_counts'}
    
    model_dict_6 = {'X':X_chi_counts,
                   'model':NN_PT_propensity_model(input_dim=len(X_chi[0]), lr=0.0001, experiment_name = 'NN_12_' + exp_name),
                   'model_name':'Simple_NN_counts'}
    
    
    model_dict_8 = {'X':X_LDA,
                   'model':LogReg_PT_propensity_model(input_dim=n_topics, lr=0.0001, experiment_name = 'LR_12_' + exp_name),
                   'model_name':'Logistic_Regression_LDA'}
    
    model_dict_9 = {'X':X_LDA,
                   'model':NN_PT_propensity_model(input_dim=n_topics, lr=0.0001, experiment_name = 'NN_12_' + exp_name),
                   'model_name':'Simple_NN_LDA'}
    
    
    # A temporary file can be added to do some precalculation, making HBERT more efficient
    d_input = None
    if len(opt.temp_file_path ) > 0:
        d = tempfile.TemporaryDirectory(prefix=opt.temp_file_path)
        d_input = d.name + '/' + exp_name
    
    model_dict_7 = {'X':X_HBERT,
                   'model':Average_BERT_propensity_model(n_it = opt.n_it,val_interval=opt.val_interval, lr=opt.lr, batch_size=opt.bs,
                                                              h_size_sent=1000, h_size_user=768, tokenize=False,
                                                              precalc_path= d_input,
                                                              experiment_name = 'avgbert' + exp_name,
                                                              seq = opt.seq, max_tokens_batch = opt.max_tokens_batch),
                                                              #preembed_size = opt.preembed_size),
                   'model_name':'avgBERT'}
    
    
    
    
    
    
    # a list of dictionaries to keep track of all models to run
    model_dicts = [model_dict_8, model_dict_9, model_dict_5, model_dict_6, model_dict_0, model_dict_1, model_dict_2, model_dict_3, model_dict_4]

    
    #
    #
    #
    ###################################
    ###################################
    #
    #
    #    
    """
    This last section runs each model for the given experiment
    
    
    
    """
    
    ## loop over the models
    stat_dicts = []
    
    if opt.no_try:
        for i, model_dict in enumerate(model_dicts):
            # only run the model if you haven't yet
            if not os.path.isfile('{}/{}.npy'.format(exp_name, model_dict['model_name'])):
                dataset.update_X(model_dict['X'])
                # fit model
                model = model_dict['model']
                _, stat_dict = train_propensity_model(model, dataset, data_test=True)
    
                stat_dicts += [stat_dict]
    
                np.save('{}/{}.npy'.format(exp_name, model_dict['model_name']), stat_dict)
    
            stat_dict = np.load('{}/{}.npy'.format(exp_name, model_dict['model_name']),allow_pickle=True).item()
            print(stat_dict)
            print(type(stat_dict))
            
            stat_dict_print = {key:stat_dict[key] for key in [k for k in stat_dict.keys() if 'P_' not in k and 'Z_' not in k and 'Y_' not in k] }
    
            print('model {}, statdict {}'.format(model_dict['model_name'], stat_dict_print))
            
        return
    
            
        
    
    
    for i, model_dict in enumerate(model_dicts):
        try:
            # only run the model if you haven't yet
            if not os.path.isfile('{}/{}.npy'.format(exp_name, model_dict['model_name'])):
                dataset.update_X(model_dict['X'])
                # fit model
                model = model_dict['model']
                _, stat_dict = train_propensity_model(model, dataset, data_test=True)
    
                stat_dicts += [stat_dict]
    
                np.save('{}/{}.npy'.format(exp_name, model_dict['model_name']), stat_dicts)
    
            stat_dict = np.load('{}/{}.npy'.format(exp_name, model_dict['model_name']), allow_pickle=True).item()
            
            stat_dict_print = {key:stat_dict[key] for key in [k for k in stat_dict.keys() if 'P_' not in k] }
    
            print('model {}, statdict {}'.format(model_dict['model_name'], stat_dict_print))
            
            
        except:
            print('model {} FAILED'.format(model_dict['model_name']))
    
    
    
    
if __name__ == '__main__':
    main()
