import numpy as np
import csv
import os
from os import path as osp


# Define the path to your CSV file
steps=5
path_1="/home/enyuzhao/code/softgym/"

path_2="1_20_config_test_"+str(steps)
state=0
path_3=f"RGBD_simple/state_{state}/coverages_final.csv"

# csv_file_path = "/home/enyuzhao/code/softgym/1_20_config_test_5/RGBD_simple/state_0/coverages_final.csv"
csv_file_path=osp.join(path_1,path_2)
csv_file_path=osp.join(csv_file_path,path_3)



# Use numpy.genfromtxt() to load the data
data = np.genfromtxt(csv_file_path, delimiter=',')

# Print the data
print(data)
print(data.shape)