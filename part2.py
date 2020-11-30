#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Oct 19 21:28:50 2020

@author: yuwenchen
"""
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%%
def normalization(x):
    return (x - min(x)) / (max(x) - min(x))

def val_normalization(x):
    global maxMinData
    return (x - maxMinData.iloc[1]) / (maxMinData.iloc[0] - maxMinData.iloc[1])

def findMaxMin(x):
    return x.max(axis=0), x.min(axis=0)

def sigmoid(x):
    sig = 1 / (1 + np.exp(-x))     # Define sigmoid function
    return sig

def zeroToOne(x):
    return np.where(x>0.5, 1 ,0)

def accuracy(predict, ans):
    length = len(ans)
    
    compList = predict == ans
    # the number of correct
    correct = sum(compList)
    return correct/length
#%%
trainSet_x = pd.DataFrame(pd.read_csv('pa2_train_X.csv'))
trainSet_y = pd.DataFrame(pd.read_csv('pa2_train_y.csv'))

valSet_x = pd.DataFrame(pd.read_csv('pa2_dev_X.csv'))
valSet_y = pd.DataFrame(pd.read_csv('pa2_dev_y.csv'))

train_numFea = trainSet_x[['Age', 'Annual_Premium', 'Vintage']]
val_numFea = valSet_x[['Age', 'Annual_Premium', 'Vintage']]

maxMinData = train_numFea.apply(findMaxMin)
norm_train_numFea = train_numFea.apply(normalization)
norm_vali_numFea = val_normalization(val_numFea)

trainSet_x = trainSet_x.drop(columns=['Age', 'Annual_Premium', 'Vintage', 'dummy'])
train_x = pd.concat([norm_train_numFea, trainSet_x], axis=1)

valSet_x = valSet_x.drop(columns=['Age', 'Annual_Premium', 'Vintage', 'dummy'])
val_x = pd.concat([norm_vali_numFea, valSet_x], axis=1)
#%%
lenFeature = len(train_x.columns)
sampleNum = len(train_x)
validNum = len(val_x)

x_train = train_x.to_numpy() # transform all data in to numpy form
x_val = val_x.to_numpy() # transform all validation set in to numpy form
y_head_train = trainSet_y.to_numpy().reshape(sampleNum) 
y_head_val = valSet_y.to_numpy().reshape(validNum)

print("data preprocessing is done, start training")
#%%
w = np.zeros(lenFeature)
b = np.array([0])

w_grad = np.zeros(lenFeature)
b_grad = np.array([0])

lr = 0.1
lamb = 0.00001

epoch = 10000

#loss_train_record = []
#loss_val_record = []
acc_train_record = []
acc_val_record = []

#best_acc = 0
#best_w = None
#%%
for i in range(epoch):
    y_train = np.sum(x_train*w, axis=1)+b # y for training set
    y_val = np.sum(x_val*w, axis=1)+b # y for validation set

    # calculate the loss (include L2 regularization)
    #loss_train = np.sum(-1*((y_head_train*np.log(sigmoid(y_train))) + ((1-y_head_train)*np.log(1-sigmoid(y_train)))))/sampleNum + lamb*np.sum(abs(w))
    #loss_val = np.sum(-1*((y_head_val*np.log(sigmoid(y_val))) + ((1-y_head_val)*np.log(1-sigmoid(y_val)))))/validNum + lamb*np.sum(abs(w))
    
    #loss_train_record.append(loss_train)
    #loss_val_record.append(loss_val)
    
    t_acc = accuracy(zeroToOne(sigmoid(y_train)) , y_head_train)
    v_acc = accuracy(zeroToOne(sigmoid(y_val)) , y_head_val)
    acc_train_record.append(t_acc)
    acc_val_record.append(v_acc)
    
    #if v_acc > best_acc:
    #    best_w = w
    
    # compute ∂L/∂w
    w_grad = ((y_head_train - sigmoid(y_train))@(x_train))/sampleNum
    # compute ∂L/∂b
    b_grad = np.sum((y_head_train - sigmoid(y_train))*(-1))/sampleNum

    w = w + lr*w_grad
    w = np.sign(w)*np.maximum(abs(w)-lamb*lr, 0)
    b = b - lr*b_grad

    if i%100 == 0: # tracing loss while training
        print('epoch:', i, 'acc_t:', t_acc, 'acc_v:', v_acc)