import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stops NUMA error
from utils.tools import read_yaml
from pt_utils.transformer_model import ActionTransformer
from pickle_wrapper import *

import numpy as np
import warnings
import torch
import torch.utils.data
import tensorflow as tf
torch.manual_seed(9)
# Check GPU
if not torch.cuda.is_available():
    warnings.warn("Cannot find GPU")
else:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
config = read_yaml("../utils/config.yaml")

# Constants
model_size = config['MODEL_SIZE']
split = 1
fold = 0
bin_path = config['MODEL_DIR']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_sz = config[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4
pos_emb = config['POS_EMB']
num_frames = 30
num_classes = 20
skel_extractor = 'openpose'


def count_parameters_pt(model):
    """
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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
            param.data = torch.tensor(weight_dict[name]).to(device)
        else:
            raise Exception(f"weight, with shape {param.data.shape}, for {name} is not found")

# Create the pt model
# n_heads doesn't impact learnable params??
# https://discuss.pytorch.org/t/what-does-increasing-number-of-heads-do-in-the-multi-head-attention/101294/4
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", layer_norm_eps=1e-6, batch_first=True)
trans_nn = torch.nn.TransformerEncoder(encoder_layer, n_layers)
model = ActionTransformer(trans_nn, d_model, num_frames, num_classes, skel_extractor, mlp_head_sz).to(device)

# Dummy input which have the same shape as the expected single vector output from openpose (1, 30, 52)
# Ensures we have set up the starting layer(s) of the AcT correctly
inputs = torch.ones(1, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                    config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']).to(device)
model(inputs)
# Count number of trainable parameters
# print(count_parameters_pt(model))

# Load keras weights OR check name of weights and their shape
weight_dict = open_pickle(f"keras_weight_dict_{model_size}.pickle")
load_weight(model, weight_dict)

# Load dummy input and save output as a numpy array
# np_input = np.load("test_array.npy")
# t_input = torch.tensor(np_input).to(torch.float).to(device)
# model.eval()
# t_output = model(t_input)
# print(t_output)
# np.save(f"pt_output_{model_size}.npy", t_output.cpu().detach().numpy())
# sys.exit()

#del from here
# print(np_input.shape)
# a = t_output
# print(torch.argmax(torch.softmax(a, dim=-1), dim=0).shape)

# import tensorflow as tf
# torch.manual_seed(9)
# loss = torch.nn.CrossEntropyLoss(reduction='sum')
# y_pred = torch.tensor([[0.05, 0.95, 0], [0.1, 0.8, 0.1]])
# y_true = torch.nn.functional.one_hot(torch.inde((2,4)),20)
# print(torch.nn.NLLLoss()(torch.log(y_pred), y_true))
# output = loss(y_pred, y_true)
# print(output.item())
# tf_loss = tf.keras.losses.CategoricalCrossentropy(from_logits=True)
# print(f"Loss with tf {tf_loss(y_true=tf.convert_to_tensor(y_true.numpy()),y_pred=tf.convert_to_tensor(y_pred.numpy()))}")

root = '/home/louis/Data/Fernandez_HAR/AcT_posenet_processed_data/'
test_x = torch.tensor(np.load(root + f"X_test_processed_split{1}_fold{1}.npy"))
test_y = torch.tensor(np.load(root + f"y_test_processed_split{1}_fold{1}.npy"))
test_dataset = torch.utils.data.TensorDataset(test_x, test_y)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=512, shuffle=False, drop_last=True)
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction='mean')

model.eval()
for bx, by in test_loader:
    # print(bx[0,:10,:10])
    label = torch.argmax(by, dim=1).to(device)
    output = model(bx.to(torch.float).to(device))
    # print(output.shape, label.shape)
    loss = loss_fn(output, label)
    print(loss)
    sys.exit()