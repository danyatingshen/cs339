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
import ols_main_simple

def main() : 
    data = ols_main_simple.define_data()
    dataset_array = ols_main_simple.read(data)
    dataset_np = np.array(dataset_array)
    # b) 
    x = dataset_np[:,0]
    t = dataset_np[:,1]
    plt.scatter(x,t) 
    w_0, w_1 = ols_main_simple.simple_coefficent_prediction(dataset_array)
    plt.plot(x, w_1*x + w_0)
    plt.xticks(np.arange(1920, 2020, 20))
    plt.yticks(np.arange(10.5, 13, 0.5))
    plt.xlabel('Year')
    plt.ylabel('Winning Time(s)')
    plt.show()
    # c) 
    t_2012 = 10.75
    t_2016 = 10.71
    t_hat_2012 = w_1*2012 + w_0
    t_hat_2016 = w_1*2016 + w_0
    t_2012_sq_error = (t_2012 - t_hat_2012) * (t_2012 - t_hat_2012)
    t_2016_sq_error = (t_2016 - t_hat_2016) * (t_2016 - t_hat_2016)
    print("2012 predicted winning is", t_hat_2012, "and squre error is:", t_2012_sq_error)
    print("2016 predicted winning is", t_hat_2016, "and squre error is:", t_2016_sq_error)

main()