#!/usr/bin/env python
# coding: utf-8

# In[173]:


## importing pandas and numpy
import numpy as np
import pandas
import math

## find the distance between two vector points
## assuming the input is (class, input1, input2)
def euclid_distance(vectorA, vectorB):
    dist = 0
    for i in range(1, len(vectorA)):
        square = (vectorA[i]-vectorB[i]) * (vectorA[i]-vectorB[i])
        dist += square
    distance = math.sqrt(dist)
    return distance

## find the distance between the training set and a given vector
## assuming the given vector = [class, input1, input2]
## assuming the training set = ([class, input1_1, input1_2], [class, input2_1, 2_2]...)
def distALL(given, train):
    all_dist = []
    for i in range(len(train)):
        all_dist.append(euclid_distance(given, train[i]))
    return all_dist

## want to add the distance to the vector values so you know which vector has which distance to given feature vector
## expected ourput will be [class, input1, input2, distance]
def add(train, dist_all):
    for i in range(len(train)):
        train[i].append(dist_all[i])
    return train

## find the k nearest neighbors
def k_neigh(data, k):
    data.sort(key = lambda x: x[3]) ## sort by the third index (distance)
    return data[0:k]

## find what the majority class is in the k nearest neighbors
def majority_class(k_neigh):
    most_class = []
    for i in range(len(k_neigh)):
        most_class.append(k_neigh[i][0])
    return np.bincount(most_class).argmax()

## find the majority class of k nearest neighbors for dataset
## if there is a tie between  indecies that appear in equal frequency, then choses the lowest index
def knn(train, test, k):
    y_hat = []
    y = []
    for i in range(len(test)): ## for every row in test
        y.append(test[i][0])
        distance = distALL(test[i], train) ## create matrix of distances for the first test value
        temp = add(train, distance) ## create temp dataset that has the distncaes in it
        k_values = k_neigh(temp, k) ## find the k nearest neighbors
        y_hat.append(majority_class(k_values)) ## append the most common value
    return y, y_hat