# Amanda Shen 
import random
import csv
import math
import sys
import operator
import numpy as np
from math import exp

#----------------------------------------------------------------------------------
# input: 
# y and y_hat list 
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
# output: 
# misclassify rate of y and y_hat
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
def misclassify_rate (y, y_hat) :
    total = 0
    error = 0
    correct = 0
    for i in range(len(y)) :
        total += 1
        if (y[i] == y_hat[i]) :
            correct += 1
        else :
            error += 1
    result = float(error)/total
    return result
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# input: 
# the name of a classifier function, -knn
# test dataset - dataset
# true y of dataset -y
# the name of a performance metric function -misclassify_rate
# argv : trainning set and k (additional)
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
# output: 
# performance of the classifier on the dataset
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
def evaluate_misclassify (knn,testing, y, misclassify_rate, k_neigbor, training) :
    y_hat = knn(training,testing,k_neigbor)
    miscal_rate = misclassify_rate(y, y_hat)
    return miscal_rate
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# input: 
# J fold -J 
# training dataset
# classifier function 
# a random seed
# additional arguments needed by the function in the previous step
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
# output: 
# performance of the classifier of folds 
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
def cross_validation (J, trainning, knn, seed, k, istraining):
    #creates a random partition of the training set into J folds
    total_length = len(trainning)
    fold = total_length/J
    random.Random(seed).shuffle(trainning)
    generator = (trainning[i:i+fold] for i in range(0, len(trainning), fold))
    master_fold_list = list(generator)
    performance_fold = list()
    performance_generalization = list()
    print(master_fold_list)
    # option 1: returns the performance of the classifier for each fold.
    # option 2: 1,(234) -> 2,(134)..., return the generalization error
    for v_index in range(len(master_fold_list)) :
        test = master_fold_list[v_index]
        #print("test",test)
        temp = master_fold_list[:v_index]+master_fold_list[v_index+1:]
        train = helper_depack(temp)
        #print("train",train)
        y = helper_find_y(test)
        #print("y",y)
        #score = evaluate_misclassify (knn,test, y, misclassify_rate, k, train)
        score = 1
        performance_fold.append(score)    
        if (istraining == True) :
            #score_2 = evaluate_misclassify (knn,test, y, misclassify_rate, k, test)
            score_2 = 2
            performance_generalization.append(score_2)
    #print(performance_fold,performance_generalization)
    return performance_fold, performance_generalization

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
#----------------------------------------------------------------------------------
#----------------------------------------------------------------------------------
# input: 
# performance of the classifier of folds lists 
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
# output:
# mean performance 
# 25th and 75th percentiles
#- - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - - - - -
def mean_performance (performance_fold,performance_generalization):
    avg = sum(performance_fold)/len(performance_fold)
    fold_75 = np.percentile(performance_fold, 75)
    fold_25 = np.percentile(performance_fold, 25)
    
    if (len(performance_generalization) != 0) :
        avg_2 = sum(performance_generalization)/len(performance_generalization)
        print(avg,fold_75,fold_25,avg_2)
        return avg, avg,fold_25,fold_75,avg_2
    else :
        return avg,fold_25,fold_75
#----------------------------------------------------------------------------------

#----------------------------------------------------------------------------------
def main () :
    y = [1,1,-1,-1]
    y_hat = [-1,1,-1,-1]
    trainning = [[1],[2],[3],[4],[5],[6]]
    knn = 0
    seed = 123
    k = 1

    #miss = misclassify_rate(y, y_hat)
    fold, gen = cross_validation (3, trainning, knn, seed, k, True)
    print(fold,gen)
    mean_performance(fold,gen)



if __name__ == '__main__':
    main()