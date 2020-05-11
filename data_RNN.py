#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 13 17:18:05 2019

@author: peterawest
"""

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

class Tokenizer():
    def __init__(self):
        self.word2ind = {'UNK':0}
        self.ind2word = ['UNK']
        self.counts = [0]
        
    def count_word(self,word):
        # count word for the purposes of vocab generation
        
        if word in self.word2ind:
            self.counts[self.word2ind[word]] = self.counts[self.word2ind[word]] + 1
        else:
            ind = len(self.ind2word)
            self.word2ind[word] = ind
            self.ind2word += [word]
            self.counts += [1]
        
        
        
        
    def prune_dictionary(self, min_count):
        self.counts[0] = min_count + 1 # save unk
        
        # filter out words that don't meet min count threshold
        inds = [v[0] for v in  filter(lambda x: x[1] > min_count, enumerate(self.counts))]
        
        new_ind2word = []
        new_word2ind = {}
        
        for i, ind in enumerate(inds):
            word = self.ind2word[ind]
            new_ind2word += [word]
            new_word2ind[word] = i
            
        self.ind2word = new_ind2word
        self.word2ind = new_word2ind
            
