import os
import sklearn.metrics
import matplotlib.pyplot as plt
import ignite.metrics
class ConfusionMatrix:
    def __init__(self, num_classes=20, labels=None):
        """
        Initialise class for confusion matrix

        Args:
            labels: Target names used for plotting
        """
        self.metric = ignite.metrics.ConfusionMatrix(num_classes=num_classes)
        self.labels = labels

    def update(self, y_pred, y_true):
        """
        Update the confusion matrix based on the model output and ground truth labels

        Args:
            y_pred: torch tensor | The model's output as logits with shape (mini_batch, num_classes)
            y_input: torch tensor | Array containing ground-truth class indices with shape (mini_batch)
        """
        self.metric.update((y_pred, y_true))

    def get_cm(self):
        """
        Getter for the confusion matrix

        Returns: confusion matrix
        """
        return self.metric.confusion_matrix

    def plot(self):
        """
        Plots the confusion matrix
        """
        self.metric.compute()
        sklearn.metrics.ConfusionMatrixDisplay(self.get_cm().numpy(), display_labels=self.labels).plot()
        fig = plt.gcf()
        fig.set_size_inches(10.8, 7.8)
        ax = plt.gca()
        ax.set_title(f"Confusion matrix")
        if not (self.labels is None):
            ax.set_xticklabels(self.labels, rotation=45, ha='right', fontsize=10)

        plt.show()
        plt.close('all')

    def save_plot(self, split, fold, path="."):
        """
        Saves the confusion matrix
        split: int | data split
        fold: int | training fold
        path: string | directory path to save the figure in
        """
        self.metric.compute()
        sklearn.metrics.ConfusionMatrixDisplay(self.get_cm().numpy(), display_labels=self.labels).plot()
        fig = plt.gcf()
        fig.set_size_inches(10.8, 7.8)
        ax = plt.gca()
        ax.set_title(f"Confusion matrix for split {split} fold {fold}")
        if not (self.labels is None):
            ax.set_xticklabels(self.labels, rotation=45, ha='right', fontsize=10)

        p = os.path.join(path, f"s_{split}_f_{fold}_confusion_matrix.jpg")
        plt.savefig(p, dpi=100, bbox_inches='tight')
        plt.close('all')
        self.metric.reset()

if __name__ == "__main__":
    import torch
    import numpy as np
    import sys
    from transformer_model import ActionTransformer
    from pickle_wrapper import *

    labels = ['standing', 'check-watch', 'cross-arms', 'scratch-head', 'sit-down', 'get-up', 'turn-around', 'walking',
              'wave1', 'boxing',
              'kicking', 'pointing', 'pick-up', 'bending', 'hands-clapping', 'wave2', 'jogging', 'jumping', 'pjump',
              'running']
    weight_dict = open_pickle(f"/home/louis/Git/AcT/compare_model/keras_weight_dict_micro.pickle")
    cm = ConfusionMatrix(20, labels)
    root = '/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/'
    test_x = torch.tensor(np.load(root + f"X_test_processed_split{1}_fold{1}.npy"))
    test_y = torch.tensor(np.load(root + f"y_test_processed_split{1}_fold{1}.npy"))
    test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=True, drop_last=True)
    loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='sum')

    encoder_layer = torch.nn.TransformerEncoderLayer(d_model=64, nhead=1, dim_feedforward=256, dropout=0.3,
                                               activation="gelu", layer_norm_eps=1e-6, batch_first=True)
    trans_nn = torch.nn.TransformerEncoder(encoder_layer, 4)
    model = ActionTransformer(trans_nn, 64, 30, 20, 'openpose', 256).to('cuda:0')
    load_weight(model, weight_dict)

    model.eval()
    for bx, by in test_loader:
        label = torch.argmax(by, dim=1)
        output = model(bx.to(torch.float).to('cuda:0'))
        prob_dist = torch.softmax(output, dim=-1)  # Apply softmax here
        labels = torch.argmax(by, dim=1).cpu().detach()
        cm.update(prob_dist.cpu().detach(), labels)
        # cm.save_plot(0,0)
    cm.save_plot(1,1)
    # cm.plot()

    # y_true = torch.tensor([0, 1, 0, 1, 0])
    # y_pred = torch.tensor([
    #     [0.2, 0.7, 0.1],
    #     [0.0, 1.0, 0.0],
    #     [1.0, 0.0, 0.0],
    #     [0.0, 1.0, 0.0],
    #     [0.0, 1.0, 0.0],
    # ])
    # cm.update(y_pred, y_true)
    # cm.update(y_pred, y_true)
    # cm.plot()
    # cm.save_plot(0,0)

