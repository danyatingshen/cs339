# Simulate N independent values of X from selected Normal distribution 
# to estimate E[g(x)]


from scipy.stats import randint
import numpy as np
import math
import pandas as pd
from matplotlib import pyplot as plt



## Input the high and low intervals and the number of draws
def draw_sample_normal_distribution(mu, std, num_draws, seed):
    np.random.seed(seed)
    return np.random.normal(mu, std, num_draws)

def function_g_of_x(drawned_sample):
    vector_cos = np.cos(drawned_sample)
    vector_g = vector_cos * math.sqrt(math.pi)
    return vector_g

# input from the specified normal distribution drraws
def estimate_expected_value(sample_drawn):
    N = sample_drawn.size
    summed = sample_drawn.sum()
    return summed/N

# simulates upto the given trials
# then outputs two columns pandas array of the trial number N and the estimated value
def trials(max_draws, seed):
    mu = 0
    std = math.sqrt(1/2)
    # do half of the max draw. 
    num_of_draws = max_draws//2
    #trials_estimated = pd.DataFrame(columns= ["N draws", "estimated value"])
    trials_estimated = list()
    for draw in range(1, num_of_draws):
        actual_draw = draw * 2
        sample_drawned = draw_sample_normal_distribution(mu, std, actual_draw, seed)
        vector_g = function_g_of_x(sample_drawned)
        estimated = estimate_expected_value(vector_g)
        trials_estimated.append([actual_draw, estimated])
    pandas_estimated = pd.DataFrame(trials_estimated, columns=["N draws", "estimated E[g(X)]"])
    return pandas_estimated

# given the number of draws and the estimated value, 
# graph them
# input should be in pandas dataframe
def graph_relation(estimated_data):
    estimated_data.plot(kind='line', x='N draws',y='estimated E[g(X)]',color='red')
    plt.savefig('part_2_b.png')

# simulates the N independent values of X and then sum as given 
# from the E[g(X)] equation
def main():
    ## mu and sigma (std) found from 2(a)
    draws = 100000
    seed = 890213
    estimated = trials(draws, seed)
    
    graph_relation(estimated)
    
    #print the last expected value
    print(estimated.tail(5))


main()

