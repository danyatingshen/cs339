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

## 1ai) function that computed euclidean distance
## find the distance between two vector points
## assuming the input is (class, input1, input2)
def euclid_distance(vectorA, vectorB):
    dist = 0
    for i in range(1, len(vectorA)):
        square = (vectorA[i]-vectorB[i]) * (vectorA[i]-vectorB[i])
        dist += square
    distance = math.sqrt(dist)
    return distance

## Maybe more faster one
def numpy_euclid(vectorA, vectorB):
    return distance.euclidean(vectorA[1:], vectorB[1:])


## 1aii) computer euclidean distance between particular feature vector and each data point in training set
## find the distance between the training set and a given vector
## assuming the given vector = [class, input1, input2]
## assuming the training set = ([class, input1_1, input1_2], [class, input2_1, 2_2]...)
def distALL(given, train):
    all_dist = []
    for i in range(len(train)):
        #all_dist.append(euclid_distance(given, train[i]))      #old array element by element distance
        all_dist.append(numpy_euclid(given, train[i]))      # different distance function
    return all_dist

## 1aiii) find the indices by first adding the distance and then finding k nearest
## want to add the distance to the vector values so you know which vector has which distance to given feature vector
## expected ourput will be [class, input1, input2, distance]
def add(train, dist_all):
    temp_added = copy.deepcopy(train) #to avoid keep indexing the distance again and again.
    for i in range(len(train)): 
        temp_added[i].append(dist_all[i])
    return temp_added

## find the k nearest neighbors
def k_neigh(data, k):
    #### We sort by the distance, which is at the end of the list
    data.sort(key = lambda x: x[-1]) ## sort by the last index (distance)
    return data[0:k]

## 1aiv) find maojrity class of the k nearest neighbor
## find what the majority class is in the k nearest neighbors
def majority_class(k_neigh):
    most_class = {}
    for i in range(len(k_neigh)):
        label = k_neigh[i][0]
        if label in most_class:
            most_class[label] = most_class[label]
        else:
            most_class[label] = 1
    # now with the LabelVotes, we sort and pick the label type with greatest number of matches
    return max(most_class.items(), key = operator.itemgetter(1))[0]

def most_frequent(List): 
    return max(set(List), key = List.count)

## 1av) find the majority class for all the points in dataset, return y and y_hat
## find the majority class of k nearest neighbors for dataset
## if there is a tie between  indecies that appear in equal frequency, then choses the lowest index
def knn(train, test, k):
    y_hat = []
    y = []
    for i in range(len(test)): ## for every test row
        y.append(test[i][0]) ## The first element. 
        distance = distALL(test[i], train) ## create matrix of distances for the first test value
        temp = add(train, distance) ## create temp dataset that has the distances in it
        k_values = k_neigh(temp, k) ## find the k nearest neighbors
        y_hat.append(majority_class(k_values)) ## append the most common value
    return y, y_hat
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
    y,y_hat = knn(training,testing,k_neigbor)
    #print(y,y_hat)
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
    fold = int(total_length/J)
    random.Random(seed).shuffle(trainning)
    generator = (trainning[i:i+fold] for i in range(0, len(trainning), fold))
    master_fold_list = list(generator)
    performance_fold = list()
    performance_generalization = list()
    # option 1: returns the performance of the classifier for each fold.
    # option 2: 1,(234) -> 2,(134)..., return the generalization error
    for v_index in range(len(master_fold_list)) :
        test = master_fold_list[v_index]
        temp = master_fold_list[:v_index]+master_fold_list[v_index+1:]
        train = helper_depack(temp)
        y = helper_find_y(test)     # the labels
        score = evaluate_misclassify (knn,test, y, misclassify_rate, k, train)
        performance_fold.append(score)    
        if (istraining == True) :
            score_2 = evaluate_misclassify (knn,test, y, misclassify_rate, k, test)
            performance_generalization.append(score_2)
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
    result = list()
    avg = sum(performance_fold)/len(performance_fold)
    fold_75 = np.percentile(performance_fold, 75)
    fold_25 = np.percentile(performance_fold, 25)
    result.append(avg)
    result.append(fold_25)
    result.append(fold_75)
    
    if (len(performance_generalization) != 0) :
        avg_2 = sum(performance_generalization)/len(performance_generalization)
        result.append(avg_2)
    return result
    

def define_data ():
    df_1 = pd.read_csv("S1test.csv", header = None)
    df_2 = pd.read_csv("S1train.csv", header = None)
    return df_1, df_2

def read_data(df):
    #df = pd.read_csv("../S1test.csv", header = None)
    data = []
    dim = np.shape(df)
    columns = df.columns
    for i in range(dim[0]): ## for every row
        temp = []
        for j in range(dim[1]): ## append every column value
            temp.append(df[columns[j]][i]) ## array of (class, input1, input2)
        data.append(temp)
    return data

def read(df):
    data = []
    dim = np.shape(df)
    for (indx, row) in df.iterrows():
        temp = row.to_list()
        data.append(temp)
    return data



def main () :
    J = 10
    k = 3
    seed = 123
    test, train = define_data()
    test_1 = read_data(test)
    another_test = read(test)
    train = read_data(train)
    
    fold, gen = cross_validation (J, train, knn, seed, k, True)
    #print(fold,gen)
    result = mean_performance(fold,gen)
    print("The following are average of trainning error, 25 , 75 percentiles and average of generalization error if applicable: ")
    for item in result :
        print(item)

