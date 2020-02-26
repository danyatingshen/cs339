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
import statistics
import ols_main_vector
import cross

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

# 1. loop through D from 1 to D+1 and nasted with lamb value from 1 to 100
# 2. 

def compare_lambda (master_trainning_x,master_trainning_t,master_testing_x,master_testing_t,D) :
    result = list()
    result.append(0)
    best_D = -1
    best_lanbda = -1
    lowest_MSE = 10000000
    for power in range(1, D+1):
        for lamb in range (1, 1001) :
            master_traiining_X = ols_main_vector.creates_predictor_matrix(master_trainning_x, power)
            w = ols_main_vector.ols_coefficent_prediction_lamda(master_traiining_X,master_trainning_t,lamb)
            t_hat = ols_main_vector.generate_predition_vector (master_testing_x,w)
            error = cross.mean_squared_error(master_testing_t, t_hat)
            result.append(error)
            if error < lowest_MSE :
                lowest_MSE = error
                best_D = power
                best_lanbda = lamb
    return best_D, best_lanbda, lowest_MSE

def main ():
    # set up dataset: 
    df = define_data()
    dataset_array = read(df)
    master_training = np.array(dataset_array)
    master_testing = np.array([[2012,10.75],[2016,10.71]])
    # Compute squared prediction errors for each of the polynomial degree
    master_trainning_x = master_training[:,0]
    master_trainning_t = master_training[:,1]
    master_testing_x = master_testing[:,0]
    master_testing_t = master_testing[:,1]
    D = len(master_trainning_t)
    J = 2
    seed = 123
    istraining = True
    lamb = 0
    best_D, best_lanbda, lowest_MSE = compare_lambda(master_trainning_x,master_trainning_t,master_testing_x,master_testing_t,D)
    print(best_D, best_lanbda, lowest_MSE)

main()