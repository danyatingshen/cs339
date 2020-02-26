# Amanda Shen 
import random
import csv
import math
import sys
import operator
import numpy as np
import pandas as pd
from math import exp
import copy 
from scipy.spatial import distance
from matplotlib import pyplot as plt
from numpy import ones

## The global global variable that we will divide by at the end and then multiply for standardization
global_max_X = 1

#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# input : 
def ols_coefficent_prediction_lamda (X,t,lamb):
    X_transpose = np.transpose(X)
    X_transpose_X = np.dot(X_transpose,X)
    row,col = X_transpose_X.shape
    I = np.identity(row)
    lamb_I = np.multiply(lamb,I)
    X_transpose_X_lamb_I = np.add(X_transpose_X,lamb_I)
    X_transpose_t = np.dot(X_transpose,t)
    try:
        w = np.linalg.solve(X_transpose_X_lamb_I,X_transpose_t)
    except:
        w= np.linalg.lstsq(X_transpose_X_lamb_I,X_transpose_t, rcond=None)[0]
    return w
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# Input: a single predictor column and a positive integer D
def creates_predictor_matrix (predictor, D):
    length = len(predictor)
    result = np.transpose(ones([length]))
    for power in range(1,D + 1) :
        new = np.power(predictor,power)
        result = np.vstack((result,new))
    result = np.transpose(result)
    return result
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# Input : true t column, x column, and D, 
# return: coefficients of the OLS polynomial function of order D
def ols (t,x,D,lamb) :
    X = creates_predictor_matrix (x, D)
    w = ols_coefficent_prediction_lamda(X,t,lamb)
    t_hat = generate_predition_vector(x,w)
    return w

def generate_predition_vector (x,w):
    y = list()
    x_index = 0
    for index in range(0,len(x)) :
        f = 0
        current = x[index] 
        for power in range(0,len(w)):
            f = f + w[power] * current**power
        y.append(f)
    y = np.array(y)
    return y
'''
def generate_predition_vector (x,w):
    y = list()
    x_index = 0
    ## Before proceeding, we need to make sure if 
    # x array is not array of array
    #if ((type(x[0]) is np.ndarray) == True):
        # if in this for-loop, it means test variable is an array of an array
        #x = x[0]  
    for index in range(0,len(x)) :
        f = 0
        #current = x[index][1]
        #x[index][1] = x[index][1]*global_max_X
        

        for power in range(0,len(w)):
            # we also need to multiply back our global max x
            current = x[index][power]*global_max_X
            f = f + w[power] * current**power
        y.append(f)
    y = np.array(y)
    return y
'''
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
def define_data ():
    df = pd.read_csv("womens100.csv", header = None)
    return df

def read(df):
    data = []
    dim = np.shape(df)
    for (indx, row) in df.iterrows():
        temp = row.to_list()
        data.append(temp)
    return data