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
from numpy import ones
import ols_main_vector

def main() : 
    data = ols_main_vector.define_data()
    dataset_array = ols_main_vector.read(data)
    dataset_np = np.array(dataset_array)
    x = dataset_np[:,0]
    t = dataset_np[:,1]
    D = 3
    lamb = 0
    w = ols_main_vector.ols(x,t,D,lamb)
    # g) 
    # plt.scatter(x,t) 
    # w_0 = w[0]
    # w_1 = w[1]
    # w_2 = w[2]
    # w_3 = w[3]
    # x = list(x)
    # y = list(np.array(ols_main_vector.helper_poly (w_0,w_1,w_2,w_3,x)))
    #print(x,y)
    #plt.plot(x,y)
    # plt.xticks(np.arange(1920, 2020, 20))
    # plt.yticks(np.arange(10.5, 13, 0.5))
    # plt.xlabel('Year')
    # plt.ylabel('Winning Time(s)')
    #plt.show()



    #h)
    # lambda_list = [1,10,100,10000]
    # for i in lambda_list :
    #     result = ols_main_vector.ols(x,t,D,i)
    #     print(result)

    #i)
    # x = np.array([[1],[2],[3]])
    # w = np.array([[2],[2],[2]])
    y = ols_main_vector.generate_predition_vector(x,w)


main()