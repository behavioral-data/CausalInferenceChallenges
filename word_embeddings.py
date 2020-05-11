#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Sep 17 10:48:27 2019

@author: peterawest
"""

from pytorch_pretrained_bert import BertTokenizer, BertModel
import torch
from torch import nn
import os

import time

class BERT_word_embedding(nn.Module):
    '''
    This embeds lists of posts as a tensor of word embeddings (n_posts x max_len x embedding_dim)
    
    '''
    def __init__(self, tokenize = True, max_tokens_batch = 10000, temp_save = None, max_len = 512,max_posts=1000):
        torch.nn.Module.__init__(self)
        
        # whether or not HBERT model will have to tokenize (or data is pretokenized)
        self.tokenize = tokenize
        
        self.max_posts = max_posts
        self.max_len = max_len
        self.max_tokens_batch = max_tokens_batch
        
        self.temp_save = temp_save
        
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        self.bert.eval()
        self.bert.to('cuda')
        
    
    
    

    def forward(self, posts, precalc = None):
#        t_temp = time.time()
        
        # first, tokenize the posts. Padded with zeros
        tokenized_posts = torch.zeros(len(posts), self.max_len).long().to('cuda')
        attention_mask = torch.zeros(len(posts), self.max_len).long().to('cuda')
        
        # must satisfy this if we are to save and load this user
        precalc_condition = precalc is not None and len(posts) > 200
        
#        print('t0: {}'.format(time.time() - t_temp))
#        t_temp = time.time()
        
        with torch.no_grad():
            
            tokenized_list = []
            max_len = -1
            
            for i, post in enumerate(posts):
                text = post
                if self.tokenize:
                    tokenized_text = self.tokenizer.tokenize(text)
                else:
                    tokenized_text = text
                    
                max_len = max([max_len, len(tokenized_text)])
                tokenized_list += [tokenized_text]    

            
            # only make input tensor as big as it needs to be
            max_len = min([max_len, self.max_len])
            tokenized_posts = torch.zeros(len(posts), max_len).long().to('cuda')
            attention_mask = torch.zeros(len(posts), max_len).long().to('cuda')
                
           
            for i, tokenized_text in enumerate(tokenized_list):
            
                # Convert token to vocabulary indices
                indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text[:max_len])
                tokenized_posts[i, :len(indexed_tokens)] = torch.tensor(indexed_tokens)
                attention_mask[i, :len(indexed_tokens)] = 1

            
            
        if precalc_condition and os.path.isfile(precalc):
            out_tensor = torch.load(precalc)
            


        else: # if this value can't be precalculated
            with torch.no_grad():        
        
                # get batches of maximum size
                batch_size = int(self.max_tokens_batch/tokenized_posts.shape[1])  
                
                # take a ceiling for number of batches needed
                n_batch = int((tokenized_posts.shape[0]-1)/batch_size) + 1
                
#                print('{} batches'.format(n_batch))
                
                out_list = []
                for j in range(n_batch):
            
                    X_batch = tokenized_posts[j*batch_size : (j+1)*batch_size]
                    mask_batch = attention_mask[j*batch_size : (j+1)*batch_size]
                    
                    out, _ = self.bert(X_batch, output_all_encoded_layers=False, attention_mask=mask_batch)
                    out_list += [out]
                
                out_tensor = torch.cat(out_list, dim=0)
                
                if precalc_condition:
                    torch.save(out_tensor, precalc)

        
        return out_tensor, attention_mask
