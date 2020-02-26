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

# compare MSE for 2012 and 2016 given predicted t_hat using trainning data 
# 1. use women entire trainning data to get w = ols_main_vector.ols_coefficent_prediction_lamda(X_train,t_train,lamb)
# 2. put the 2012 matrix 's x into function generate_predition_vector (x,w) and get t_hat
# 3. calculate MSE for each poly's result 
def compare (master_trainning_x,master_trainning_t,master_testing_x,master_testing_t,D,lamb) :
    result = list()
    result.append(0)
    for power in range(1, D+1):
        master_traiining_X = ols_main_vector.creates_predictor_matrix(master_trainning_x, power)
        w = ols_main_vector.ols_coefficent_prediction_lamda(master_traiining_X,master_trainning_t,lamb)
        t_hat = ols_main_vector.generate_predition_vector (master_testing_x,w)
        error = cross.mean_squared_error(master_testing_t, t_hat)
        result.append(error)
    return result

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
    result = compare(master_trainning_x,master_trainning_t,master_testing_x,master_testing_t,D,lamb)
    print(result)

main()