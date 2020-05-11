#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 11:05:38 2019

@author: peterawest
"""

import random
from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
import numpy as np
import math
import os

import time

from word_embeddings import BERT_word_embedding

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to('cuda')
sig = torch.nn.Sigmoid()


from torch import nn
MAX_LENGTH = 100
sm_1 = torch.nn.Softmax(dim=1)
sm = torch.nn.Softmax()


max_posts = 1000


# calculates position embeddings given seq_len and n_dim
def position_embeddings(n_dim, seq_len):

    
    inds = torch.arange(n_dim).expand([seq_len,n_dim]).float().to('cuda')
    pos = torch.arange(seq_len).view(-1,1).expand([seq_len, n_dim]).float().to('cuda')
    
    position_embeddings = (inds > n_dim/2).float()*torch.sin(pos/(1000**(2*inds/n_dim)) ) + (inds <= n_dim/2).float()*torch.cos(pos/(1000**(2*inds/n_dim)) ) 
    return position_embeddings

class AttnDot_batch(nn.Module):
    def __init__(self, hidden_size):
        torch.nn.Module.__init__(self)
        self.hidden_size = hidden_size
        #!!! initialize this a better way
        self.query_vec = torch.nn.parameter.Parameter(torch.randn(hidden_size).to('cuda'))

    def forward(self, input, attn_mask = None):
        
        inp_shape = input.shape
        input = input.view(inp_shape[0]*inp_shape[1], inp_shape[2])
        
        attn_logits = input.matmul(self.query_vec.view(1,-1,1)).view(inp_shape[0],inp_shape[1])
        
        if attn_mask is not None: 

            logits = attn_logits - 100000*(attn_mask == 0).float()
            
        else:
            logits = attn_logits

        attn_weights = sm(logits)
       
        weights =  attn_weights*((attn_weights == attn_weights).float()) + 0*((attn_weights != attn_weights).float())   #attn_weights[attn_weights != attn_weights] = 0

        #assert((weights.sum(dim=1) < 1.0001).all() and (weights.sum(dim=1) > 0.999).all())

        input = input.view(inp_shape[0], inp_shape[1], inp_shape[2])

        weighted = weights.unsqueeze(2)*input
        rep = weighted.sum(dim=1)
        return rep

from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to('cuda')
sig = torch.nn.Sigmoid()


from pytorch_pretrained_bert import BertTokenizer, BertModel

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert = BertModel.from_pretrained('bert-base-uncased')
bert.eval()
bert.to('cuda')
sig = torch.nn.Sigmoid()


class Hierarchical_BERT(nn.Module):
    def __init__(self, h_size_sent, h_size_user, tokenize = True, max_posts=1000, 
                 max_len = 512, max_tokens_batch = 10000, temp_save = None, PW = False,
                 seq = False):
        torch.nn.Module.__init__(self)
        
        self.h_size_sent = h_size_sent
        self.h_size_user = h_size_user
        
        self.linear0 = torch.nn.Linear(768,h_size_sent).to('cuda')
        
        self.attn_post = AttnDot_batch(h_size_sent)
        
        self.linear1 = torch.nn.Linear(h_size_sent,h_size_user).to('cuda')
        
        self.attn_user = AttnDot_batch(h_size_user)
        
        
        self.PW = PW
        
        if self.PW:
            self.classifier_0 = torch.nn.Linear(h_size_user + 1,50).to('cuda')
            self.classifier_1 = torch.nn.Linear(50,1).to('cuda')
        else:
            self.classifier = torch.nn.Linear(h_size_user,1).to('cuda')
        
        # whether or not HBERT model will have to tokenize (or data is pretokenized)
        self.tokenize = tokenize
        
        
        self.max_posts = max_posts
        self.max_len = max_len
        self.max_tokens_batch = max_tokens_batch
        
        self.temp_save = temp_save
        
        self.bert_embedding = BERT_word_embedding(tokenize = self.tokenize, max_tokens_batch=self.max_tokens_batch, 
                                                  temp_save = self.temp_save, max_len=self.max_len,max_posts=self.max_posts)
        
        self.seq = seq
        if self.seq:
            self.seq_w = torch.nn.parameter.Parameter(torch.randn(1).to('cuda'))
    
    def preembed(self, post_list):
        
        flat = []
        for posts in post_list:
            flat += posts
            
        embedding_tensor, attention_mask = self.bert_embedding( flat ) 


        lens = [0] + [len(posts) for posts in post_list]
        
        for i in range(1,len(lens)):
            lens[i] = lens[i] + lens[i-1]
        
        
        embedding_list = [ embedding_tensor[lens[i]:lens[i+1]] for i in range(len(lens) - 1)]
        
        attention_list = [ attention_mask[lens[i]:lens[i+1]] for i in range(len(lens) - 1)]
        
        
        return list(zip(embedding_list, attention_list))
        
        
    
    def forward(self, X, precalc = None, preembedded = None):
#        post_reps = []
#        
#        for post in posts:
#            post_reps += [self.forward_post(post).view(1,-1)]
#
#        post_tensor = torch.cat(post_reps,dim=0).unsqueeze(0)
        
        if self.PW:
            posts, treatment = X[0], X[1]
        else:
            posts= X
        
        
        
        if preembedded is not None:
            embedding_tensor, attention_mask = preembedded
            
#            print('e_ten = {}'.format(embedding_tensor.shape[0]))
#            print('posts = {}'.format(len(posts)))
            #assert(embedding_tensor.shape[0]  == len(posts))
        else:
            embedding_tensor, attention_mask = self.bert_embedding( posts   ) 
        
        
        # if sequential, add sine waves underneath!
        if self.seq:
            embedding_tensor, attention_mask = self.bert_embedding( posts   ) 
            temp = self.linear1( self.attn_post( self.linear0( embedding_tensor  ), attn_mask = attention_mask)).unsqueeze(0)
            
            
            temp = self.forward_post_reps(posts, precalc)
            post_tensor = temp + self.seq_w*position_embeddings(self.h_size_user, temp.shape[1]).unsqueeze(0)
        else:

            post_tensor = self.linear1( self.attn_post( self.linear0( embedding_tensor  ), attn_mask = attention_mask)).unsqueeze(0)
        
        
        
        #assert(post_tensor.shape[1] == len(X))
        

        user_rep = self.attn_user(post_tensor)

        
        if self.PW:
            treat_tensor = torch.tensor([treatment]).float().to('cuda')
            classifier_input = torch.cat([user_rep,treat_tensor])
            
            out = sig(self.classifier_1( sig(self.classifier_0(classifier_input) ) ).view(1) )
        else:
            classifier_input = user_rep
            out = sig(self.classifier(classifier_input).view(1))
        
        

        return out
        


class Average_BERT(nn.Module):
    def __init__(self, h_size_sent, h_size_user, tokenize = True, max_posts=1000, 
                 max_len = 512, max_tokens_batch = 10000, temp_save = None, PW = False):
        torch.nn.Module.__init__(self)

        
        self.linear0 = torch.nn.Linear(768,h_size_user.to('cuda'))     
        #self.linear1 = torch.nn.Linear(h_size_sent,h_size_user).to('cuda')
        
        
        self.PW = PW
        
        if self.PW:
            self.classifier_0 = torch.nn.Linear(h_size_user + 1,50).to('cuda')
            self.classifier_1 = torch.nn.Linear(50,1).to('cuda')
        else:
            self.classifier = torch.nn.Linear(h_size_user,1).to('cuda')
        
        # whether or not HBERT model will have to tokenize (or data is pretokenized)
        self.tokenize = tokenize
        
        
        self.max_posts = max_posts
        self.max_len = max_len
        self.max_tokens_batch = max_tokens_batch
        
        self.temp_save = temp_save
        
        self.bert_embedding = BERT_word_embedding(tokenize = self.tokenize, max_tokens_batch=self.max_tokens_batch, 
                                                  temp_save = self.temp_save, max_len=self.max_len,max_posts=self.max_posts)
        

    
    def preembed(self, post_list):
        
        flat = []
        for posts in post_list:
            flat += posts
            
        embedding_tensor, attention_mask = self.bert_embedding( flat ) 


        lens = [0] + [len(posts) for posts in post_list]
        
        for i in range(1,len(lens)):
            lens[i] = lens[i] + lens[i-1]
        
        
        embedding_list = [ embedding_tensor[lens[i]:lens[i+1]] for i in range(len(lens) - 1)]
        
        attention_list = [ attention_mask[lens[i]:lens[i+1]] for i in range(len(lens) - 1)]
        
        
        return list(zip(embedding_list, attention_list))
        
        
    
    def forward(self, X, precalc = None, preembedded = None):
#        post_reps = []
#        
#        for post in posts:
#            post_reps += [self.forward_post(post).view(1,-1)]
#
#        post_tensor = torch.cat(post_reps,dim=0).unsqueeze(0)
        
        if self.PW:
            posts, treatment = X[0], X[1]
        else:
            posts= X
        
        
        
        if preembedded is not None:
            embedding_tensor, attention_mask = preembedded
            
#            print('e_ten = {}'.format(embedding_tensor.shape[0]))
#            print('posts = {}'.format(len(posts)))
            #assert(embedding_tensor.shape[0]  == len(posts))
        else:
            embedding_tensor, attention_mask = self.bert_embedding( posts   ) 
        
        for i in attention_mask.shape[0]:
            embedding_tensor[:, attention_mask == 0] = 0. # zero out omitted vectors
            
        post_tensor = self.linear0( embedding_tensor  ).mean(dim=1)

        user_rep = post_tensor.mean(dim = 0)

        
        if self.PW:
            treat_tensor = torch.tensor([treatment]).float().to('cuda')
            classifier_input = torch.cat([user_rep,treat_tensor])
            
            out = sig(self.classifier_1( sig(self.classifier_0(classifier_input) ) ).view(1) )
        else:
            classifier_input = user_rep
            out = sig(self.classifier(classifier_input).view(1))
        
        

        return out
        

class Hierarchical_BERT_propensity_model():
    
    def __init__(self, n_it = 100000, val_interval = 1000, batch_size = 1, lr = 0.001, h_size_sent = 768, h_size_user = 768, tokenize = True, max_tokens_batch = 10000,
                 precalc_path = None, experiment_name = 'hbert', PW = False, seq = False, preembed_size = 10,
                 agg = 'attn'):
        
        if agg is 'attn':
            self.hb = Hierarchical_BERT(h_size_sent=h_size_sent, h_size_user=h_size_user, tokenize = tokenize, max_tokens_batch = max_tokens_batch, PW = PW, seq=seq)
        elif agg is 'avg':
            self.hb = Average_BERT(h_size_sent=h_size_sent, h_size_user=h_size_user, tokenize = tokenize, max_tokens_batch = max_tokens_batch, PW = PW)
        else:
            assert(False)
        
        
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.n_it = n_it # number of training iterations
        self.lr = lr
        self.precalc_path= precalc_path
        assert(val_interval < n_it)
        print('val_interval is: {}'.format(val_interval))
        
        self.experiment_name = experiment_name
        self.preembed_size = preembed_size
    
    def fit(self, dataset, verbose = True):
        self.learning_curve = []
        opt = torch.optim.Adam(self.hb.parameters(), lr=self.lr)
        opt.zero_grad()
        
        preembed_size = self.preembed_size
        
        min_val_loss = None
        
        t_start = time.time()
        

        
        iterations = []
        posts_list = []
        ### loop over sgd iterations
        for it_, (posts_,label_,_, ind_) in enumerate(dataset.train_epoch(size=self.n_it, include_ind = True)):
            
#            if it_ % 100 == 0:
#                print('at it_ {}, time is {}'.format(it_,time.time() - t_start))
            
#            ex = random.randint(0, len(X_train) - 1)
#            posts = X_train[ex]
#            label = Z_train[ex]
            
            
            
            
            if len(posts_) > 1000:
                print('posts is too long ({})'.format(len(posts_)))
                random.shuffle(posts_)
                posts_ = posts_[:1000]
                

            iterations += [(it_, posts_,label_, ind_)]
            posts_list += [posts_]
            
#            print('it_:{}'.format(it_))
            # if 
            if ((it_ + 1) % preembed_size) == 0:
#                print('it_ = {}'.format(it_))
#                print('preembed time')
                preembedded = self.hb.preembed(posts_list)
                
                for j, iteration in enumerate(iterations):
                    it, posts,label, ind = iteration

                    
                    # if precalculating BERT reps, do this here
                    if False:# self.precalc_path is not None:
                        precalc = self.precalc_path + 'ex_{}'.format(ind)
                        logit = self.hb.forward(posts, precalc = precalc)
                    else:
                        logit = self.hb.forward(posts, preembedded = preembedded[j])
                        #logit = self.hb.forward(posts)
                    loss = -(float(label)*torch.log(logit) + float(1-label)*torch.log(1-logit))
                    
                    if math.isnan(loss.item()):
                        
                        print('Is NAN! iteration: {}',format(it))
                        
                        self.lr = self.lr/2.
                        print('reloading model, dropping lr to {}'.format(self.lr))
                        self.load_best()
                        opt = torch.optim.Adam(self.hb.parameters(), lr=self.lr)
                        opt.zero_grad()
                        
                        continue
                        
                        
                    
                    (loss/self.batch_size).backward()
                    
                    if ((it % self.batch_size) == 0) and it > 0:
                        torch.cuda.empty_cache()
                        opt.step()
                        opt.zero_grad()
                        torch.cuda.empty_cache()
                        
                    if ((it) % self.val_interval) == 0:
                        print('time to validate')
                        t_val_start = time.time()
                        
                        torch.cuda.empty_cache()
                        with torch.no_grad():
                            #print('starting validation')
                            val_loss = 0
                            for val_i, (posts, label,_)  in enumerate(dataset.valid_epoch()):
        #                        posts = X_val[val_i]
        #                        label = Z_val[val_i]
        
                                logit = self.hb.forward(posts)
                                loss = -(float(label)*torch.log(logit) + float((1-label))*torch.log(1-logit))
        
                                val_loss += float(loss.item())
                            self.learning_curve += [val_loss]
                            #print('val_loss: {}'.format(val_loss))
                            if min_val_loss is None or val_loss < min_val_loss:
                                min_val_loss = val_loss
                                
                                self.save_best()
        #                        torch.save(self.hb.state_dict(),'best.pt')
                            elif val_loss > 1.5*min_val_loss:
                                break
                            
                        if verbose:
                            print('it: {}, val_loss: {}, time: {}'.format(it ,val_loss, time.time() - t_start))
                            print('time to val: {}'.format(time.time() - t_val_start))
        #                    print('time = {}'.format(time.time() - t_start))
                            
                iterations = []
                posts_list = []  
        self.load_best()
        #        self.hb.load_state_dict(torch.load('best.pt'))
     
      
    def load_best(self):
        self.hb.load_state_dict(torch.load('{}_best.pt'.format(self.experiment_name)))
        
    def save_best(self):
        torch.save(self.hb.state_dict(),'{}_best.pt'.format(self.experiment_name))
    
    def score(self, X):
        n_ex = len(X)
        scores = np.zeros(n_ex)
        
        
        for i in range(len(X)):
            data_in = X[i]

            with torch.no_grad():
                logit = self.hb.forward(data_in).item()
            scores[i] = logit
        return scores
                    

    
