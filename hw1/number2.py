# Testing and getting #2

## importing pandas and numpy
import numpy as np
import pandas as pd
import math
import matplotlib.pyplot as plt # for visualization
from PIL import Image
## Importing our knn program
import knnpy

import time



# sets up the necessary data for training and testing the pixel files
# Also will save the pictures

def data_setup():
    ## reading in the dataframe
    tr_5 = pd.read_csv("Train5.csv", header = 0)

    tr_9 = pd.read_csv("Train9.csv", header = 0)

    te_5 = pd.read_csv("Test5.csv", header = 0)
    te_9 = pd.read_csv("Test9.csv", header = 0)

    # Reshape each row to 28 x 28 matrix to check picture
    # for trial 5
    nump_tr_5 = tr_5.to_numpy()
    mat_tr_5 = nump_tr_5[1].reshape(28, 28)

    # for trial 9
    nump_tr_9 = tr_9.to_numpy()
    mat_tr_9 = nump_tr_9[1].reshape(28, 28)

    # for test 5
    nump_te_5 = te_5.to_numpy()
    mat_te_5 = nump_te_5[1].reshape(28, 28)

    # for test 9
    nump_te_9 = te_9.to_numpy()
    mat_te_9 = nump_te_9[1].reshape(28, 28)

    ## Check for the picture
    Image.fromarray(np.uint8(mat_tr_5), "L").save("trial5", "JPEG")
    Image.fromarray(np.uint8(mat_tr_9), "L").save("trial9", "JPEG")

    Image.fromarray(np.uint8(mat_te_5), "L").save("test5", "JPEG")
    Image.fromarray(np.uint8(mat_te_9), "L").save("test9", "JPEG")
    
    ## After checking the pictures, now we add the label values and get ready for KNN
    tr_5.insert(0, "label", 5)
    tr_9.insert(0, "label", 9)
    te_5.insert(0, "label", 5)
    te_9.insert(0, "label", 9)

    training_set = pd.concat([tr_5, tr_9], axis =0)
    test_set = pd.concat([te_5, te_9], axis =0)

    train = knnpy.read(training_set)
    test = knnpy.read(test_set)

    
    return train, test






def get_result (train, test, k):
    J = 10
    seed = 123
    istraining = True

    validation_error,trainning_error,generalization_error = knnpy.cross_validation (J, train, knnpy.knn, seed, k, istraining)
    #print(trainning_error,generalization_error)
    result = knnpy.mean_performance(validation_error,trainning_error,generalization_error)
    print("Performance of each fold (average validation error): ",result[0])
    print("validation error 25 and 75 percentiles: ",result[1],result[2])
    if (istraining):
        print("Average of Trainning error: ",result[3])
        print("Average of generalization error: ",result[4])
    
    return result
    
def define_data ():
    df_1 = pd.read_csv("S1test.csv", header = None)
    df_2 = pd.read_csv("S1train.csv", header = None)
    return df_1, df_2

def main():
    t0 = time.time()

    #train_data, test_data = data_setup()
    test_data, train = define_data()    # the shorter data for debugging
    train_data = knnpy.read(train)
    ## Generalization or not
    generalizaiton = True

    ## We make data frame and keep k value and the mean error after
    ### We then set the num of seeds here and loop
    max_loop = len(train_data)//2
    error_tracker = pd.DataFrame(columns = ["k", "mean error", "25 percentile", "75 percentile", "avg generalization error"])
    
    # Create a Pandas Excel writer using XlsxWriter as the engine.
    writer = pd.ExcelWriter('mean_error.xlsx', engine='xlsxwriter')

    for i in range(20):
    #for i in range(3):
        k = 1 + i*2
        result = get_result(train_data, test_data, k)
        if (len(result) > 3):       # if the avg generalization error is present
            error_tracker.loc[len(error_tracker)] = [k, result[0], result[1], result[2], result[3]]
        else:
            error_tracker.loc[len(error_tracker)] = [k, result[0], result[1], result[2], "NA"]

    error_tracker.to_excel(writer, index = False)
    writer.save()
    t1 = time.time()

    total = t1-t0
    print("THIS HOW LONG IT TOOK: ", total)



    

if __name__ == '__main__':
    
    main()



