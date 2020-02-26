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
    target_npCheck =  (type(target) is np.__name__)
    prediction_npCheck = (type(prediction) is np.__name__)
    
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
    MSE_list_validation = list()
    MSE_list_trainning = list()
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
        #print(X_train,X_test)
        X_test_x = test[:,2]
        X_train_x = train[:,2]
        # put train to ols_coefficent_prediction_lamda (X,t,lamb) and get w
        w = ols_main_vector.ols_coefficent_prediction_lamda(X_train,t_train,lamb)
        # generate t_hat by calling generate_predition_vector(x,w) where w from previous step and x is test set
        t_hat = ols_main_vector.generate_predition_vector(X_test_x,w)
        # use MSE function to caluclate (t,t_hat) for error rate
        socre = mean_squared_error(t_test,t_hat)
        MSE_list_validation.append(socre)
        if (istraining) :
            t_hat_trainning_error = ols_main_vector.generate_predition_vector(X_train_x,w)
            # use MSE function to caluclate (t,t_hat) for error rate
            socre_2 = mean_squared_error(t_train,t_hat_trainning_error)
            MSE_list_trainning.append(socre_2)
    # keep track of each loop's MSE and return the mean and standard deviation across folds of the MSE.
    result = list()
    if istraining == False:
        std = statistics.stdev(MSE_list_validation)
        mean = statistics.mean(MSE_list_validation)
        result.append(mean)
        result.append(std)
        return result
    else:
        std = statistics.stdev(MSE_list_validation)
        mean = statistics.mean(MSE_list_validation)
        std_2 = statistics.stdev(MSE_list_trainning)
        mean_2 = statistics.mean(MSE_list_trainning)
        result.append(mean)
        result.append(std)
        result.append(mean_2)
        result.append(std_2)
        return result


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


## TO fix: the default parameter for D = N (number of data points)
def best_poly_cross_validation (t, x, D = None, K = 2,seed = 1,istraining = False):
    if (D == None) :
        D = len(t)
    validation_error_mean = list()
    validation_error_std = list()
    trainning_error_mean = list()
    trainning_error_std = list()
    lamb = 0        
    lowest_order = 0    
    lowest_mean = 9999999    
    for power in range(1, D+1):
        X = ols_main_vector.creates_predictor_matrix(x, power)
        result = cross_validation (t, X, K,seed,lamb,istraining)

        validation_mean = result[0]
        validation_sdt = result[1]

        validation_error_mean.append(validation_mean)
        validation_error_std.append(validation_sdt)

        if(lowest_mean > validation_mean):
            lowest_mean = validation_mean
            lowest_order = power

        if (istraining == True):
            trainning_mean = result[2]
            trainning_std = result[3]
            trainning_error_mean.append(trainning_mean)
            trainning_error_std.append(trainning_std)
    
    if (istraining == True):        
        return validation_error_mean, validation_error_std, lowest_mean, lowest_order, trainning_error_mean, trainning_error_std
    else:
        return validation_error_mean, validation_error_std, lowest_mean, lowest_order

            
def optimal_order(t, x, K):
    D = len(x) -1
    lamb = 0
    poly_Means, poly_Stds, lowest_mean, lowest_order,tr_error_Means, tr_error_Stds = best_poly_cross_validation(t, x, D, K, 797897, True)
    plot_index = list(range(1,D+1))
    fig = plt.figure()
    plt.errorbar(plot_index,poly_Means, yerr = poly_Stds)
    plt.show()
    Excel_ified(poly_Means, poly_Stds, "synthdata")

#put the data to excel
def Excel_ified(Means, Stds, name = "error rate"):
    pd_excel = pd.DataFrame(Means)
    pd_excel["Above 1 SD"] = np.add(Means, Stds)
    pd_excel["Below 1 SD"] = np.subtract(Means, Stds)
    # making sure our index starts from 1 not 0
    pd_excel.index = pd_excel.index + 1     
    writer = pd.ExcelWriter(str(name) + '.xlsx', engine='xlsxwriter')
    pd_excel.to_excel(writer, index = True)
    writer.save()

# We loop through the independent variables in dataset
# input: X variables. No Y
def standardization_X(X_predictors):
   
    np_predictor = np.array(X_predictors)
    ols_main_vector.global_max_X = np.amax(np_predictor)
    return (np.divide(np_predictor, ols_main_vector.global_max_X))



def main() :
    #dataset_np = np.array([[4,3,2],[8,2,7],[15,3,9],[4,5,6],[9,5,2],[7,9,4]])
    data = ols_main_vector.define_data()
    dataset_array = ols_main_vector.read(data)
    dataset_np = np.array(dataset_array)
    x = dataset_np[:,0]
    t = dataset_np[:,1]
    D = 10
    J = 50
    seed = 123
    istraining = True
    lamb = 10
    
    #cross_validation (t,x,J,seed,lamb, istraining)
    #print(best_poly_cross_validation (t, x, lamb, D, J, seed, istraining))
    #print(best_poly_cross_validation (t, x, 0, D, J, seed, istraining))
    #meanerror = womentesting(dataset_np, 1)
    #print(meanerror)
    x_standardized = standardization_X(x)


    optimal_order(t,x_standardized, J)
main()