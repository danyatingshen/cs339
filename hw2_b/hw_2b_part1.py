
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

# (a)plug in to py(y)
def calculate_Y_equal_y (y,lamb):
    return round((math.exp(-lamb)* lamb**y)/math.factorial(y),6)
# (b)plug in to py(y) for each k
def calculate_Y_lessequal_k (k,lamb):
    sum = 0
    for y in range(0,k+1):
        sum += calculate_Y_equal_y(y,lamb)
    return sum

# (c)inputs λ and n -> n independent random variables in Possion 
    # loop n time: 
    # 1. ganerate random uniform dis u : set.seed(124)
    # 2. if u ≤ F_Y (k) = P_y(Y <= k), add k to list
    #    if not, increase k+= 1, test again k = 0....i
def trail (lamb,N):
    result = list()
    for round in range(0,N):
        k = 0
        u = random.uniform(0, 1)
        while u > calculate_Y_lessequal_k(k,lamb):
            p = calculate_Y_lessequal_k(k,lamb)
            k += 1
            #print(u > calculate_Y_lessequal_k(k,lamb),calculate_Y_lessequal_k(k,lamb),k)
        result.append(k)
    return result

def calculate_mean (list):
    return statistics.mean(list)
def calculate_var (list):
    return statistics.variance(list)
# (e): E[Y] = λ
def relationship (lamb, maxN) :
    x = list()
    y_mean = list()
    y_variance = list()
    for N in range(2,maxN):
        sample = trail (lamb,N)
        x.append(N)
        y_mean.append(calculate_mean(sample))
        y_variance.append(calculate_var(sample))
    plt.plot(x, y_mean)
    plt.xlabel('N')
    plt.ylabel('Mean')
    plt.title(str(lamb)+"as lambda Mean")
    plt.show()
    plt.plot(x, y_variance)
    plt.xlabel('N')
    plt.ylabel('Variance')
    plt.title(str(lamb)+"as lambda Varaince")
    plt.show()
# ?: how large for N from (e)

def main():
    y = 3.5
    lamb = 20
    random.seed(631)
    #relationship(20,100)
    #relationship(45,200)
    relationship(80,200)
    
main()
