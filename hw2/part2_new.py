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

def mean_squared_error(target, prediction):
    #numpy check to see if need to be transformed
    target_npCheck =  (type(target).__module__ == np.__name__)
    prediction_npCheck = (type(prediction).__module__ == np.__name__)
    
    # check and transform the data to numpy
    if (target_npCheck and prediction_npCheck == False):
        target = np.array(target)
        prediction = np.array(prediction)
    
    MSE = np.square(target - prediction).mean()
    return MSE
    
# split to J different fold, 1 -> 234
def cross_validation (t,X,J,seed,lamb, istraining) :
    # concate t and X and named it X' and go to next step
    X_prime = np.column_stack((t,X))
    X_prime = X_prime.tolist()
    trainning = X_prime
    # random split X' to J fold -> A list of list of list
    total_length = len(trainning)
    fold = int(total_length/J)
    random.Random(seed).shuffle(trainning)
    generator = (trainning[i:i+fold] for i in range(0, len(trainning), fold))
    master_fold_list = list(generator)
    MSE_list = list()
    # loop on index of each fold: 
    for v_index in range(len(master_fold_list)) :
        # set current train and test
        test = np.array(master_fold_list[v_index])
        temp = master_fold_list[:v_index]+master_fold_list[v_index+1:]
        train = np.array(helper_depack(temp))
        t_test = test[:,0]
        t_train = train[:,0]
        X_test = test[:,1:]
        X_train = train[:,1:]
        # put train to ols_coefficent_prediction_lamda (X,t,lamb) and get w
        w = ols_main_vector.ols_coefficent_prediction_lamda(X_train,t_train,lamb)
        # generate t_hat by calling generate_predition_vector(x,w) where w from previous step and x is test set
        t_hat = ols_main_vector.generate_predition_vector(X_test,w)
        # use MSE function to caluclate (t,t_hat) for error rate
        socre = mean_squared_error(t_test,t_hat)
        MSE_list.append(socre)
    # keep track of each loop's MSE and return the mean and standard deviation across folds of the MSE.
    std = statistics.stdev(MSE_list)
    mean = statistics.mean(MSE_list)
    print(std,mean)
    return mean,std
    
def helper_find_y (test) :
    y = list()
    for i in range(len(test)):
        y.append(test[i][0])
    return y

def helper_depack (train) :
    result = list()
    for i in train :
        for j in i :
            result.append(j)
    return result

def best_poly_cross_validation (t, x, D = len(t), K = 2,seed = 1,istraining = False) :
    poly_Means = list()
    poly_Stds = list()
    tr_error_Means = list()
    tr_error_Stds = list()

    lamb = 0        # doesn't consider the ridge regression here (?)
    lowest_order = 0    # will keep track the polynomial value with lowest MSE avg
    lowest_mean = 9999999     #the lowest MSE avg from the lowest_order polynomial value
    if (istraining == False):
        for power in range(0, D+1):
            predictor_matrix = ols_main_vector.creates_predictor_matrix(x, power)
            temp_mean, temp_std = cross_validation (t,x, K, seed,lamb, istraining)
            poly_Means.append(temp_mean)
            poly_Stds.append(temp_std)
            if(lowest_mean > temp_mean):
                lowest_mean = temp_mean
                lowest_order = power
        return poly_Means, poly_Stds, lowest_mean, lowest_order
    else:
        for power in range(0, D+1):
            predictor_matrix = ols_main_vector.creates_predictor_matrix(x, power)
            temp_mean, temp_std, temp_error = cross_validation (t,x, K, seed,lamb, istraining)
            poly_Means.append(temp_mean)
            poly_Stds.append(temp_std)
            tr_error_Means.append(statistics.mean(temp_error))
            tr_error_Stds.append(statistics.mean(temp_error))
            if(lowest_mean > temp_mean):
                lowest_mean = temp_mean
                lowest_order = power
        return poly_Means, poly_Stds, lowest_mean, lowest_order, tr_error_Means, tr_error_Stds

            
    #  loop through each possibility of D from 0 to D
    #     use current index = D
    #     call creates_predictor_matrix to create X according to D
    #     call cross validation and get mean and standard deviation for each order 
    #     if istrianning == True, 
    #         evaluate_misclassify (knn,train, y_train, misclassify_rate, k, train) for trainning erorr
    
#    find and return the polynomial order with the lowest average cross-validation MSE


def main() :
    dataset_np = np.array([[4,3,2],[8,2,7],[15,3,9],[4,5,6],[9,5,2],[7,9,4]])
    x = dataset_np[:,1:]
    t = dataset_np[:,0]
    D = 3
    J = 3
    seed = 123
    istraining = True
    lamb = 0
    cross_validation (t,x,J,seed,lamb, istraining)


main()