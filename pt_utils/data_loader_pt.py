#!/usr/bin/env python3.8
"""
Pytorch Dataloader for the mpose dataset
"""
import sys
from torch.utils.data import Dataset, DataLoader
from mpose import MPOSE
import yaml
from sklearn.model_selection import train_test_split

labels = {  # 20 Classes
    "standing": 0,
    "check-watch": 1,
    "cross-arms": 2,
    "scratch-head": 3,
    "sit-down": 4,
    "get-up": 5,
    "turn": 6,
    "walk": 7,
    "wave1": 8,
    "box": 9,
    "kick": 10,
    "point": 11,
    "pick-up": 12,
    "bend": 13,
    "hands-clap": 14,
    "wave2": 15,
    "jog": 16,
    "jump": 17,
    "pjump": 18,
    "run": 19}

def get_datasets(config_path, split=1, fold=0):
    """
    Loads mpose using code from original AcT dataloader
    Args:
        config_path: string or path | config file
        split: int | split
        fold: int | fold
    Returns: torch.utils.data.Dataset | Dataset objects for train, test and val
    """

    class MposeDataset(Dataset):
        def __init__(self, x, y):
            """
            Creates a Dataset class for the input values
            Args:
                x: np array | contains features
                y: np array | contains labels
            """
            self.x = x
            self.y = y

        def __len__(self):
            return self.y.shape[0]

        def __getitem__(self, idx):
            return self.x[idx], self.y[idx]

    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataset = config["DATASET"]
    d = MPOSE(pose_extractor=dataset,
              split=split,
              preprocess=None,
              velocities=True,
              remove_zip=False,
              verbose=False)

    d.reduce_keypoints()
    d.scale_and_center()
    d.remove_confidence()
    d.flatten_features()

    # Obtain np array of the train/test features and labels. The training test needs to be split into train/val
    X_tv, y_tv, X_test, y_test = d.get_data()
    X_train, X_val, y_train, y_val = train_test_split(X_tv, y_tv,
                                                      test_size=config['VAL_SIZE'],
                                                      random_state=config['SEEDS'][fold],
                                                      stratify=y_tv)

    return MposeDataset(X_train, y_train), MposeDataset(X_test, y_test), MposeDataset(X_val, y_val)



if __name__ == '__main__':
    train_dataset, test_dataset, val_dataset = get_datasets("../utils/config.yaml")
    for x, y in DataLoader(train_dataset, batch_size=64):
        print(y.shape)