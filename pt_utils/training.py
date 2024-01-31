# Imports

import sys
import warnings
warnings.filterwarnings("ignore", message="y_pred contains classes not in y_true")
warnings.filterwarnings("ignore", message="enable_nested_tensor is True, but self.use_nested_tensor is False because encoder_layer.self_attn.num_heads is odd")

from transformer_model import *
from scheduler import CustomSchedule
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.tools import Logger
from confusion_matrix import ConfusionMatrix as _cm

import pickle_wrapper as _pw
import matplotlib.pyplot as plt
import yaml
import numpy as np
import torch.utils.data
import os
import time
import statistics

def check_dir(path):
    """
    Check if directory exists. If not, create the directory
    Args:
        path: string | path to directory
    Returns: None
    """
    if not os.path.exists(path):
        os.makedirs(path)

def load_weight(model, weight_dict):
    """
    Loads the weight to the target model
    Args:
        model: torch model
        weight_dict: dictionary | contains the names of each weight and their values

    Returns: None
    """
    for name, param in model.named_parameters():
        if name in weight_dict.keys():
            param.data = torch.tensor(weight_dict[name]).to("cuda:0")
        else:
            raise Exception(f"weight, with shape {param.data.shape}, for {name} is not found")

class Trainer:
    """
    Trains the AcT network in Pytorch
    """

    def __init__(self, conf_path=None, split=1, fold=0):
        """
        Args:
            conf_path: string | path to config file
            split: int | starting data split
            fold: int | starting data fold
        """
        # Starting fold/split
        self.split = split
        self.fold = fold

        # Define constants
        if conf_path is None:
            CONFIG_PATH = "../utils/config.yaml"
        else:
            CONFIG_PATH = conf_path

        with open(CONFIG_PATH, "r") as file:
            config = yaml.safe_load(file)
        self.model_sz = config["MODEL_SIZE"]
        self.n_heads = config[self.model_sz]["N_HEADS"]
        self.d_model = config[self.model_sz]["EMBED_DIM"]  # Size of embedded input. dv = 64, made constant according to paper
        self.dropout = config[self.model_sz]["DROPOUT"]
        self.n_layers = config[self.model_sz]['N_LAYERS']
        self.mlp_head_sz = config[self.model_sz]["MLP"]  # Output size of the ff layer prior the classification layer
        self.num_frames = config["openpose"]["FRAMES"]
        self.num_classes = config["openpose"]["CLASSES"]
        self.d_ff = 4*self.d_model  # Output size of the first non-linear layer in the transformer encoder
        assert self.d_model == 64*self.n_heads
        self.skel_extractor = "openpose"

        self.weight_dict = _pw.open_pickle(f"keras_{self.model_sz}_init.pickle")
        # self.weight_dict = _pw.open_pickle(f"AcT_1_0_micro.pickle")
        self.SCHEDULER = config["SCHEDULER"]
        self.N_EPOCHS = config["N_EPOCHS"]
        self.BATCH_SIZE = config["BATCH_SIZE"]
        self.WEIGHT_DECAY = config["WEIGHT_DECAY"]
        self.WARMUP_PERC = config["WARMUP_PERC"]
        self.STEP_PERC = config["STEP_PERC"]
        self.N_FOLD = config["FOLDS"]
        self.N_SPLITS = config["SPLITS"]
        # Check GPU
        if not torch.cuda.is_available():
            warnings.warn("Cannot find GPU")
            self.DEVICE = "cpu"
        else:
            self.DEVICE = "cuda:0"

        # Paths to save results
        now = datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
        self.weights_path = f"/home/louis/Data/Fernandez_HAR/AcT_pt/{self.model_sz}_{now}/weights/"
        self.log_path = f"/home/louis/Data/Fernandez_HAR/AcT_pt/{self.model_sz}_{now}/logs/"
        self.plots_path = f"/home/louis/Data/Fernandez_HAR/AcT_pt/{self.model_sz}_{now}/plots/"
        check_dir(self.weights_path)
        check_dir(self.log_path)
        check_dir(self.plots_path)
        log_file = os.path.join(self.log_path, 'log.txt')
        self.logger = Logger(log_file)

    def get_data(self):
        """
        Loads pre-processed data into a pt dataloader
        Returns: None
        """
        root = '/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/'
        train_x = torch.tensor(np.load(root + f"X_train_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        train_y = torch.tensor(np.load(root + f"y_train_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        test_x = torch.tensor(np.load(root + f"X_test_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        test_y = torch.tensor(np.load(root + f"y_test_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        val_x = torch.tensor(np.load(root + f"X_val_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        val_y = torch.tensor(np.load(root + f"y_val_processed_split{self.split}_fold{self.fold}.npy")).to(torch.float32)
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        self.training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.BATCH_SIZE, shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.BATCH_SIZE)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.BATCH_SIZE)

    def get_model(self):
        """
        Creates the AcT model given the params in config file
        Returns: None
        """
        # Build network
        transformer = TransformerEncoder(d_model=self.d_model, num_heads=self.n_heads,
                                         d_ff=self.d_ff, dropout=self.dropout, n_layers=self.n_layers)
        self.model = ActionTransformer(transformer, self.d_model, self.num_frames, self.num_classes,
                                       self.skel_extractor, self.mlp_head_sz).to(self.DEVICE)

        # Keras code have train len incl val,
        train_len = len(self.training_loader.dataset) + len(self.val_loader.dataset)
        self.train_steps = np.ceil(float(train_len) / self.BATCH_SIZE)

        optimiser = torch.optim.Adam(self.model.parameters(), weight_decay=self.WEIGHT_DECAY, eps=1e-07)
        self.lr = CustomSchedule(d_model=self.d_model, optimizer=optimiser,
                                 n_warmup_steps=self.train_steps * self.N_EPOCHS * self.WARMUP_PERC,
                                 decay_step=self.train_steps * self.N_EPOCHS * self.STEP_PERC)

        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1)


    def main(self):
        # train: uses train/val, eval: uses test
        for split in range(1, self.N_SPLITS + 1):
            self.logger.save_log(f"----- Start Split {split} at {time.time()} ----\n")
            self.split = split
            acc_list = []
            bal_acc_list = []

            for fold in range(self.N_FOLD):
                self.logger.save_log(f"----- Start Fold {fold+1} at {time.time()} ----")
                self.fold = fold
                self.get_data()
                self.get_model()
                load_weight(self.model, self.weight_dict)  # Use the initialised Keras weights
                weight_path = self.train()
                acc, bal_acc = self.eval(weight_path)
                acc_list.append(acc)
                bal_acc_list.append(bal_acc)
                self.logger.save_log(f"Accuracy Test: {acc} <> Balanced Accuracy: {bal_acc}\n")

            self.logger.save_log(f"---- Split {split} ----")
            self.logger.save_log(f"Accuracy mean: {statistics.mean(acc_list)}")
            self.logger.save_log(f"Accuracy std: {statistics.pstdev(acc_list)}")
            self.logger.save_log(f"Balanced Accuracy mean: {statistics.mean(bal_acc_list)}")
            self.logger.save_log(f"Balanced Accuracy std: {statistics.pstdev(bal_acc_list)}")
            self.logger.save_log(f"---- Split {split} ----\n")

    def train(self):
        """
        Training function
        Saves the best model for each epoch based on the validation dataset

        Returns: string | path to the best model for across all epochs
        """
        max_acc = 0.0
        epoch_loss_train = []
        epoch_loss_val = []
        for epoch in range(self.N_EPOCHS):
            train_loss = 0
            self.model.train()
            for batch_x, batch_y in self.training_loader:
                self.lr.zero_grad()
                output = self.model(batch_x.to(self.DEVICE))
                loss = self.loss_fn(output, torch.argmax(batch_y.to(self.DEVICE), dim=1))
                loss.backward()
                self.lr.step_and_update_lr()
                train_loss += loss.detach().cpu().item() * batch_x.shape[0]
            epoch_loss_train.append(train_loss/len(self.training_loader.dataset))

            # Perform test in between epoch
            val_loss = 0
            self.model.eval()
            acc_list = []
            for batch_x, batch_y in self.val_loader:
                output = self.model(batch_x.to(self.DEVICE))
                pred = torch.argmax(torch.softmax(output, dim=1), dim=1)
                labels = torch.argmax(batch_y, dim=1).to(self.DEVICE)
                val_loss += self.loss_fn(output, labels).detach().cpu().item() * batch_x.shape[0]

                # Obtain accuracy metrics for test dataset
                acc_list.append(accuracy_score(labels.detach().cpu(), pred.detach().cpu()))
            epoch_loss_val.append(val_loss/len(self.val_loader.dataset))

            # AcT Keras used val accuracy to choose best weight. Compare this accuracy against the max val_acc
            curr_acc = statistics.mean(acc_list)
            if curr_acc > max_acc:
                # Save the best weights and the split/fold
                max_acc = curr_acc
                save_path = os.path.join(self.weights_path, f"s_{self.split}_f{self.fold}_best.pt")
                torch.save(self.model.state_dict(), save_path)

        # Plot and save loss information for this epoch
        self.fold_loss(epoch_loss_train, epoch_loss_val)
        return save_path

    def eval(self, weight_path=None):
        """
        Evaluate model on the test dataset
        Args:
            weight_path: string | path to model weight
        Returns: tuple | (accuracy and balanced accuracy)
        """
        if not (weight_path is None):
            self.model.load_state_dict(torch.load(weight_path))
        acc_list = []
        bal_acc_list = []
        conf_matr = _cm(self.num_classes, labels=self.config["LABELS"])

        self.model.eval()
        for batch_x, batch_y in self.test_loader:
            output = self.model(batch_x.to(self.DEVICE))
            labels = torch.argmax(batch_y, dim=1).detach().cpu()

            prob_dist = torch.softmax(output, dim=1)  # Normalise output
            conf_matr.update(prob_dist.detach().cpu(), labels)  # Update confusion matrix
            pred = torch.argmax(prob_dist, dim=1).detach().cpu()  # Obtain prediction as class indices
            acc_list.append(accuracy_score(labels, pred))  # Calculate accuracy
            bal_acc_list.append(balanced_accuracy_score(labels, pred))  # Calculate balanced accuracy

        # Save/calculate metrics
        conf_matr.save_plot(self.split, self.fold, self.plots_path)
        acc = statistics.mean(acc_list)
        bal_acc = statistics.mean(bal_acc_list)

        return acc, bal_acc

    def fold_loss(self, train_loss, val_loss):
        """
        Plots and saves how the epoch loss evolves for this split/fold
        Args:
            train_loss: list | training loss throughout the epochs
            val_loss: list | testing loss throughout the epochs
        Returns: None
        """
        x = np.linspace(1, self.N_EPOCHS, self.N_EPOCHS)

        fig, ax1 = plt.subplots()
        fig.suptitle(f"Loss for split {self.split} fold {self.fold}")
        fig.set_size_inches(10.8, 7.2)
        ax1.plot(x, train_loss, color='r', label='training')
        ax1.set_ylabel("Loss", color='r', fontsize=14)
        ax1.plot(x, val_loss, color='b', label='validation', alpha=0.8)
        plt.legend()
        plt.savefig(self.plots_path + f"s_{self.split}_f_{self.fold}.jpg", dpi=100)

        _pw.save_pickle(self.plots_path + f"s_{self.split}_f_{self.fold}_train_loss.pickle", train_loss)
        _pw.save_pickle(self.plots_path + f"s_{self.split}_f_{self.fold}_test_loss.pickle", val_loss)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()
    #TODO: feed skeleton data from IKEA-ASM