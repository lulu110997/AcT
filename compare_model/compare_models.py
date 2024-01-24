"""
Compares the keras and pytorch models using the first 2 data points from
/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/X_test_processed_split1_fold0.npy
"""
import numpy as np
import yaml

TEST_DATA_PATH = "/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/X_test_processed_split1_fold0.npy"
TEST_LABEL_PATH = "/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/y_test_processed_split1_fold0.npy"
with open("/home/louis/Git/AcT/utils/config.yaml", "r") as file:
    config = yaml.safe_load(file)
model_size = config["MODEL_SIZE"]


def save_test_array():
    # test_data = np.load(TEST_DATA_PATH)[:2, :, :]
    # np.save("test_array.npy", test_data)
    test_label = np.load(TEST_LABEL_PATH)
    print(np.argmax(test_label[:2,:],1))
    # np.save("test_label.npy", test_label)

def compare_output_arrays():
    pt_array = np.load(f"pt_output_{model_size}.npy")
    tf_array = np.load(f"tf_output_{model_size}.npy")
    array_diff = (pt_array - tf_array)
    print(array_diff)

compare_output_arrays()