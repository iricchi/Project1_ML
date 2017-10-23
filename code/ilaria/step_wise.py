import sys

my_path = r'/home/ilaria/Scrivania/Machine_Learning/Project_1/Project1_ML'
sys.path.insert(0,my_path + r'/code/COMMON')

import numpy as np 
import matplotlib.pyplot as plt
from proj1_helpers import load_csv_data, predict_labels 
from implementations import *
from outliers import handle_outliers
from labels import idx_2labels
from standard import standardize
from split_data import split_data
sys.path.insert(0,my_path + r'/code/ilaria')
from i_cross_validation_methods import *

# Subdived the X features space in single features
all_features = np.genfromtxt(my_path + r'/data/train.csv', delimiter=",", dtype=str, max_rows = 1)[2:]
# converting array in list in order to simplify the adding of features
all_features = list(all_features)

features = []
for i in range(len(all_features)):
    features.append((i,all_features[i]))

def results_r2_stepwise(list_r2_adj,indices_features):
    print("R2 asjusted values:")
    
    for i in range(len(list_r2_adj)):
        print(list_r2_adj[i])
    print("-------------------------------------------------------")
    print("Number of features chosen:", len(indices_features))
    print("\n")
    print("Indices of features chosen: ", indices_features)
    

