# Amanda Shen 
import random
import csv
import math
import sys
import operator
import numpy as np
import pandas as pd
from sklearn import preprocessing
from math import exp


def misclassify (y, y_hat) :
    total = 0
    error = 0
    correct = 0
    for i in len(y) :
        total += 1
        if (y[i] == y_hat[i]) :
            correct += 1
        else :
            error += 1
    result = error/total
    return result


def main () :
    y = [1,1,-1,-1]
    y_hat = [-1,1,-1,-1]
    miss = misclassify(y, y_hat)

    print("knn:running", miss)

if __name__ == '__main__':
    main()