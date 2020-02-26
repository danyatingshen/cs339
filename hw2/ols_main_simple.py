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
#----------------------------------------------------------------------------------------------
#----------------------------------------------------------------------------------------------
# input nx2 array for (x_n, t_n)
# Need to compute average of x, t, xt, x^2
def simple_coefficent_prediction (dataset) :
    x_mean,t_mean,xt_mean,x_sq_mean = helper_average(dataset)
    #print(x_mean,t_mean,xt_mean,x_sq_mean)
    w_0,w_1 = helper_coefficent(x_mean,t_mean,xt_mean,x_sq_mean)
    return w_0, w_1

def helper_coefficent (x_mean,t_mean,xt_mean,x_sq_mean) :
    w_1 = (xt_mean - x_mean * t_mean)/(x_sq_mean -x_mean * x_mean)
    w_0 = t_mean - w_1 * x_mean
    return w_0,w_1

def helper_average (dataset) :
    n = 0
    sum_x = 0
    sum_t = 0
    sum_xt = 0
    sum_x_squre = 0
    for line in dataset :
        n += 1
        sum_x += line[0]
        sum_t += line[1]
        sum_xt += (line[0] * line[1])
        sum_x_squre += (line[0] * line[0])
    x_mean = float(sum_x)/n
    t_mean = float(sum_t)/n
    xt_mean = float(sum_xt)/n
    x_sq_mean = float(sum_x_squre)/n
    return x_mean,t_mean,xt_mean,x_sq_mean
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
