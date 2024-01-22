# Imports
import sys

from transformer_model import ActionTransformer
from scheduler import CustomSchedule
from datetime import datetime
from sklearn.metrics import balanced_accuracy_score, accuracy_score
from utils.tools import Logger

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
            self.config = yaml.safe_load(file)
        self.model_sz = self.config["MODEL_SIZE"]
        self.n_heads = self.config[self.model_sz]["N_HEADS"]
        self.d_model = self.config[self.model_sz]["EMBED_DIM"]  # Size of embedded input. dv = 64, made constant according to paper
        self.dropout = self.config[self.model_sz]["DROPOUT"]
        self.n_layers = self.config[self.model_sz]['N_LAYERS']
        self.mlp_head_sz = self.config[self.model_sz]["MLP"]  # Output size of the ff layer prior the classification layer
        self.num_frames = self.config["openpose"]["FRAMES"]
        self.num_classes = self.config["openpose"]["CLASSES"]
        self.d_ff = 4*self.d_model  # Output size of the first non-linear layer in the transformer encoder
        assert self.d_model == 64*self.n_heads
        self.skel_extractor = "openpose"

        self.SCHEDULER = self.config["SCHEDULER"]
        self.N_EPOCHS = self.config["N_EPOCHS"]
        self.BATCH_SIZE = self.config["BATCH_SIZE"]
        self.WEIGHT_DECAY = self.config["WEIGHT_DECAY"]
        self.WARMUP_PERC = self.config["WARMUP_PERC"]
        self.STEP_PERC = self.config["STEP_PERC"]
        self.LR_MULT = self.config["LR_MULT"]
        self.N_FOLD = self.config["FOLDS"]
        self.N_SPLITS = self.config["SPLITS"]
        self.DEVICE = torch.device("cuda:0")

        self.weights_path = f"/home/louis/Data/Fernandez_HAR/AcT_pt/{self.model_sz}/weights/"
        self.log_path = f"/home/louis/Data/Fernandez_HAR/AcT_pt/{self.model_sz}/logs/"
        check_dir(self.weights_path)
        check_dir(self.log_path)
        now = datetime.now()
        log_file = os.path.join(self.log_path, now.strftime("%Y_%m_%d_%H_%M_%S")+'.txt')
        self.logger = Logger(log_file)

    def get_data(self):
        """
        Loads pre-processed data into a pt dataloader
        Returns: None
        """
        root = '/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/'
        train_x = torch.tensor(np.load(root + f"X_train_processed_split{self.split}_fold{self.fold}.npy"))
        train_y = torch.tensor(np.load(root + f"y_train_processed_split{self.split}_fold{self.fold}.npy"))
        test_x = torch.tensor(np.load(root + f"X_test_processed_split{self.split}_fold{self.fold}.npy"))
        test_y = torch.tensor(np.load(root + f"y_test_processed_split{self.split}_fold{self.fold}.npy"))
        val_x = torch.tensor(np.load(root + f"X_val_processed_split{self.split}_fold{self.fold}.npy"))
        val_y = torch.tensor(np.load(root + f"y_val_processed_split{self.split}_fold{self.fold}.npy"))
        train_dataset = torch.utils.data.TensorDataset(train_x, train_y)
        test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
        val_dataset = torch.utils.data.TensorDataset(val_x, val_y)
        self.training_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.config['BATCH_SIZE'],
                                                           shuffle=True)
        self.test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.config['BATCH_SIZE'],
                                                       shuffle=True)
        self.val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.config['BATCH_SIZE'],
                                                      shuffle=True)

    def get_model(self):
        """
        Creates the AcT model given the params in config file
        Returns: None
        """
        # Build network
        encoder_layer = torch.nn.TransformerEncoderLayer(d_model=self.d_model, nhead=self.n_heads,
                                                         dim_feedforward=self.d_ff, dropout=self.dropout,
                                                         activation="gelu", layer_norm_eps=1e-6, batch_first=True)
        transformer = torch.nn.TransformerEncoder(encoder_layer, self.n_layers)
        self.model = ActionTransformer(transformer, self.d_model, self.num_frames, self.num_classes,
                                       self.skel_extractor, self.mlp_head_sz).to(self.DEVICE)

        self.train_steps = np.ceil(float(len(self.training_loader.dataset)) / self.config['BATCH_SIZE'])
        self.test_steps = np.ceil(float(len(self.test_loader.dataset)) / self.config['BATCH_SIZE'])

        # https://stackoverflow.com/questions/69576720/implementing-custom-learning-rate-scheduler-in-pytorch
        # optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])
        optimiser = torch.optim.AdamW(self.model.parameters(), lr=1e-4, betas=(0.9, 0.999), eps=1e-07,
                                      weight_decay=self.config['WEIGHT_DECAY'])
        self.lr = CustomSchedule(d_model=self.d_model, optimizer=optimiser,
                            n_warmup_steps=self.train_steps * self.config['N_EPOCHS'] * self.config['WARMUP_PERC'],
                            lr_mul=self.train_steps * self.config['N_EPOCHS'] * self.config['STEP_PERC'])

        # loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
        # https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
        # Might need to implement my own loss to copy keras's categorical cross entropy loss
        # https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/7
        # https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/7
        # or not?
        # https://discuss.pytorch.org/t/cant-replicate-keras-categoricalcrossentropy-with-pytorch/146747
        self.loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum")


    def main(self):
        # Get data, train (uses train/val), eval (uses test)
        for split in range(1, self.N_SPLITS + 1):
            acc_list = []
            bal_acc_list = []
            self.logger.save_log(f"----- Start Split {split} at {time.time()} ----\n")
            self.split = split

            for fold in range(self.N_FOLD):
                self.logger.save_log(f"----- Start Fold {fold} at {time.time()} ----\n")
                self.fold = fold
                self.get_data()
                self.get_model()
                weights_path = self.train()
                acc, bal_acc = self.eval(weights_path)
                acc_list.append(acc[0])
                bal_acc_list.append(bal_acc[0])
                self.logger.save_log(f"Accuracy mean: {acc[0]}")
                self.logger.save_log(f"Accuracy std: {acc[1]}")
                self.logger.save_log(f"Balanced Accuracy mean: {bal_acc[0]}")
                self.logger.save_log(f"Balanced Accuracy std: {bal_acc[1]}")

            self.logger.save_log(f"\n---- Split {split} ----")
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
        max_bal_acc = 0.0

        for epoch in range(self.N_EPOCHS):
            self.model.train()
            for batch_x, batch_y in self.training_loader:
                # Call scheduler at the right time
                # https://stackoverflow.com/questions/69576720/implementing-custom-learning-rate-scheduler-in-pytorch
                # TODO: model.zero_grad() or optimiser.zero_grad()? Call at epoch or batch
                # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
                self.lr.zero_grad()
                batch_x = batch_x.to(torch.float)
                batch_y = batch_y.to(torch.float)
                output = self.model(batch_x.to(self.DEVICE))
                loss = self.loss_fn(output, batch_y.to(self.DEVICE))
                loss.backward()
                self.lr.step_and_update_lr()

            # Perform test in between epoch
            self.model.eval()
            bal_acc_list = []
            for batch_x, batch_y in self.val_loader:
                batch_x = batch_x.to(torch.float)
                batch_y = batch_y.to(torch.float)
                output = self.model(batch_x.to(self.DEVICE))
                pred = torch.argmax(torch.softmax(output, dim=-1), dim=1)  # Apply softmax here
                labels = torch.argmax(batch_y, dim=1)
                bal_acc_list.append(balanced_accuracy_score(labels.cpu().detach(), pred.cpu().detach()))
            bal_acc = statistics.mean(bal_acc_list)
            if bal_acc > max_bal_acc:
                # Save the weights, split, fold, epoch and batch info
                max_bal_acc = bal_acc
                save_path = os.path.join(self.weights_path, f"s_{self.split}_f{self.fold}_bal_acc_{max_bal_acc}.pt")
                torch.save(self.model.state_dict(), save_path)

        return save_path

    def eval(self, weight_path=None):
        """
        Evaluate model on the test dataset
        Args:
            weight_path: string | path to weight
        Returns: tuple | (accuracy and accuracy std dev) and (balanced accuracy, balanced accuracy std dev)
        """
        self.model.load_state_dict(torch.load(weight_path))
        acc_list = []
        bal_acc_list = []

        self.model.eval()
        for batch_x, batch_y in self.test_loader:
            batch_x = batch_x.to(torch.float)
            batch_y = batch_y.to(torch.float)
            output = self.model(batch_x.to(self.DEVICE))
            pred = torch.argmax(torch.softmax(output, dim=-1), dim=1)  # Apply softmax here
            labels = torch.argmax(batch_y, dim=1)
            acc_list.append(accuracy_score(labels.cpu().detach(), pred.cpu().detach()))
            bal_acc_list.append(balanced_accuracy_score(labels.cpu().detach(), pred.cpu().detach()))
        acc = statistics.mean(acc_list)
        acc_std_dev = statistics.pstdev(acc_list)
        bal_acc = statistics.mean(bal_acc_list)
        bal_acc_std_dev = statistics.pstdev(bal_acc_list)

        return (acc, acc_std_dev), (bal_acc, bal_acc_std_dev)

if __name__ == '__main__':
    trainer = Trainer()
    trainer.main()