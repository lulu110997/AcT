"""
Compares the keras and pytorch models using the first 2 data points from
/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/X_test_processed_split1_fold0.npy
"""
import numpy as np
import torch
import tensorflow as tf

TEST_DATA_PATH = "/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/X_test_processed_split1_fold0.npy"

def save_test_array():
    test_data = np.load(TEST_DATA_PATH)[:2, :, :]
    np.save("test_array.npy", test_data)

def compare_output_arrays():
    pt_array = np.load("pt_output.npy")
    tf_array = np.load("tf_output.npy")
    array_diff = (pt_array - tf_array)
    print(array_diff)

compare_output_arrays()