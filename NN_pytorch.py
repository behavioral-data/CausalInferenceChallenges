#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 16 09:02:36 2019

@author: peterawest
"""

import random
import torch
import numpy as np
import math


from torch import nn
MAX_LENGTH = 100
sm = torch.nn.Softmax(dim=0)



sig = torch.nn.Sigmoid()




sig = torch.nn.Sigmoid()


class NN_PT_2(nn.Module):
    def __init__(self, input_dim, hidden_dim = 10, PW = False):
        super(NN_PT_2, self).__init__()
        
        self.PW = PW
        
        if self.PW:
            input_dim = input_dim + 1
        
        
        self.linear0 = torch.nn.Linear(input_dim, hidden_dim)
        self.linear1 = torch.nn.Linear(hidden_dim, 1)
        
        
        
    def forward(self, x):
        if self.PW:
            input_tens = torch.cat([torch.tensor(x[0]).float(),torch.tensor([x[1]]).float()] )
            outputs = self.linear1(  sig( self.linear0( input_tens)))
        else:
            outputs = self.linear1(  sig( self.linear0( torch.tensor(x).float() ) ) )
        return sig(outputs)
    
class NN_PT_3(nn.Module):
    def __init__(self, input_dim, hidden_dim_0 = 50, hidden_dim_1 = 10, PW = False):
        super(NN_PT_3, self).__init__()
        
        self.PW = PW
        
        if self.PW:
            input_dim = input_dim + 1
        
        
        self.linear0 = torch.nn.Linear(input_dim, hidden_dim_0)
        self.linear1 = torch.nn.Linear(hidden_dim_0, hidden_dim_1)
        self.linear2 = torch.nn.Linear(hidden_dim_1, 1)
        
        
        
    def forward(self, x):
        if self.PW:
            input_tens = torch.cat([torch.tensor(x[0]).float(),torch.tensor([x[1]]).float()] )
            outputs = self.linear2 ( sig( self.linear1(  sig( self.linear0( input_tens)))))
        else:
            outputs = self.linear1(  sig( self.linear0( torch.tensor(x).float() ) ) )
        return sig(outputs)
        

class NN_PT_propensity_model():
    
    def __init__(self, n_it = 100000, val_interval = 1000, batch_size = 1, lr = 0.001, input_dim = 768, layers = 2,
                 experiment_name = 'NN', PW = False):
        
        if layers ==2:
            self.model = NN_PT_2(input_dim, PW = PW)
        elif layers == 3:
            self.model = NN_PT_3(input_dim, PW = PW)
        else: 
            assert(False)
        
        self.batch_size = batch_size
        self.val_interval = val_interval
        self.n_it = n_it # number of training iterations
        self.lr = lr
        assert(val_interval < n_it)
        
        self.experiment_name = experiment_name
    
    def fit(self, dataset):
        self.learning_curve = []
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        opt.zero_grad()
        
        min_val_loss = None
        
        
        ### loop over sgd iterations
        for it, (x,label,_) in enumerate(dataset.train_epoch(size=self.n_it)):
#            ex = random.randint(0, len(X_train) - 1)
#            x = X_train[ex]
#            label = Z_train[ex]
            
            
            logit = self.model.forward(x)
            loss = -(float(label)*torch.log(logit) + float(1-label)*torch.log(1-logit))
            
            if math.isnan(loss.item()):
                
                print('Is NAN! iteration: {}',format(it))
                
                self.lr = self.lr/2.
                print('reloading model, dropping lr to {}'.format(self.lr))
                self.load_best()
                opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
                opt.zero_grad()
                
                continue
            
            loss.backward()
            
            if (it % self.batch_size == 0) and it > 0:
                torch.cuda.empty_cache()
                opt.step()
                opt.zero_grad()
                torch.cuda.empty_cache()
                
            if (it % self.val_interval == 0):
                torch.cuda.empty_cache()
                with torch.no_grad():
                    #print('starting validation')
                    val_loss = 0
                    for val_i, (x, label,_)  in enumerate(dataset.valid_epoch()):
#                        x = X_val[val_i]
#                        label = Z_val[val_i]

                        logit = self.model.forward(x)
                        loss = -(float(label)*torch.log(logit) + float((1-label))*torch.log(1-logit))

                        val_loss += float(loss.item())
                    self.learning_curve += [val_loss]
                    #print('val_loss: {}'.format(val_loss))
                    if min_val_loss is None or val_loss < min_val_loss:
                        min_val_loss = val_loss
                        self.save_best()
#                        torch.save(self.model.state_dict(),'best.pt')
                    elif val_loss > 1.5*min_val_loss:
                        break
                    
                
        self.load_best()
#        self.model.load_state_dict(torch.load('best.pt'))

    def load_best(self):
        self.model.load_state_dict(torch.load('{}_best.pt'.format(self.experiment_name)))
        
    def save_best(self):
        torch.save(self.model.state_dict(),'{}_best.pt'.format(self.experiment_name))

        
    def score(self, X):
        n_ex = len(X)
        scores = np.zeros(n_ex)
        
        
        for i in range(len(X)):
            x = X[i]

            with torch.no_grad():
                logit = self.model.forward(x).item()
            scores[i] = logit
        return scores