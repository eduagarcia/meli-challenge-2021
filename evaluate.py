import numpy as np
import pandas as pd
import os
import sys

from utils import read_numpy

def rps(np_predicted, np_true):
    return np.sum(np.square(np.cumsum(np_predicted, axis=1) - np.cumsum(np_true, axis=1)), axis=1).mean()  

def evaluate_rps(predict, ground_truth):
    np_predicted = read_numpy(predict)
    np_true = read_numpy(ground_truth)
    
    return rps(np_predicted, np_true)

if __name__ == "__main__":
    predict = sys.argv[-2]
    ground_truth = sys.argv[-1]
    print('predict:', predict)
    print('ground_truth:', ground_truth)
    print(evaluate_rps(predict, ground_truth))