#!/usr/bin/env python
# coding: utf-8

# In[173]:


## importing pandas and numpy
import numpy as np
import pandas
import math


# In[159]:


## reading in the dataframe
df = pandas.read_csv("~/STAT339/S1test.csv", header = None)
# print (np.shape(df))
print(df)


# In[165]:


## save dataset as array [(class1, input1_1, input1_2), (class2, input2_1, input2_2), ...]
def read_data(dataset):
    data = []
    dim = np.shape(df)
    for i in range(dim[0]): ## for every row
        temp = []
        for j in range(dim[1]): ## append every column value
            temp.append(df[j][i]) ## array of (class, input1, input2)
        data.append(temp)
        return data


# In[185]:


## find the distance between two vector points
## assuming the input is (class, input1, input2)
def euclid_distance(vectorA, vectorB):
    dist = 0
    for i in range(1, len(vectorA)):
        square = (vectorA[i]-vectorB[i]) * (vectorA[i]-vectorB[i])
        dist += square
    distance = math.sqrt(dist)
    return distance


# In[179]:


## testing euclid_distance()
train = ((1, 2, 3))
test = ((1, 4, 5))
print(test, train)
print(euclid_distance(test, train))


# In[186]:


## find the distance between the training set and a given vector
## assuming the given vector = [class, input1, input2]
## assuming the training set = ([class, input1_1, input1_2], [class, input2_1, 2_2]...)
def distALL(given, train):
    all_dist = []
    for i in range(len(train)):
        all_dist.append(euclid_distance(given, train[i]))
    return all_dist


# In[187]:


## testing distALL()
given = (0, 1, 2)
test = ([0, 1, 2], [0, 2, 3], [0, 3, 4])
print(given, test)
print(distALL(given, test))


# In[194]:


## want to add the distance to the vector values so you know which vector has which distance to given feature vector
## expected ourput will be [class, input1, input2, distance]
def add(train, dist_all):
    for i in range(len(train)):
        train[i].append(dist_all[i])
    return train


# In[203]:


## testing add()
train = [[0, 0, 0], [1, 1, 1], [2, 2, 2]]
dist_all = (5, 6, 7)
print(train, dist_all)
print(add(train, dist_all))


# In[216]:


## find the k nearest neighbors
def k_neigh(data, k):
    data.sort(key = lambda x: x[3]) ## sort by the third index (distance)
    return data[0:k]


# In[217]:


## testing k_neigh()
train = [[0, 0, 0], [1, 1, 1], [2, 2, 2], [4, 5, 6], [3, 6, 8], [1, 3, 4], [1, 5, 9], [33, 567, 9]]
dist_all = (9, 16, 7, 8, 0, 2, 5, 5)
test = add(train, dist_all)
print("before", train)
print("after", k_neigh(train, 3))


# In[222]:


## find what the majority class is in the k nearest neighbors
def majority_class(k_neigh):
    most_class = []
    for i in range(len(k_neigh)):
        most_class.append(k_neigh[i][0])
    return np.bincount(most_class).argmax()


# In[225]:


## testing majority_class()
train = [[0, 0, 0], [0, 1, 1], [10, 2, 2], [0, 5, 6], [0, 6, 8], [0, 3, 4], [0, 5, 9], [0, 567, 9]]
print(majority_class(train))


# In[229]:


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


# In[239]:


## testing knn
train = [[0, 0, 0], [2, 1, 1], [2, 2, 2], [3, 3, 3]]
test = [[4, 4, 4], [5, 5, 5], [6, 6, 6], [7, 7, 7]]
print(knn(train, test, 3))


# In[ ]:





# In[ ]:




