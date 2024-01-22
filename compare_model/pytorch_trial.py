import os; os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stops NUMA error
from utils.tools import read_yaml
from pt_utils.transformer_model import ActionTransformer
from pickle_wrapper import *

import numpy as np
import torch
import tensorflow as tf

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


# Check GPU
if not torch.cuda.is_available():
    raise Exception("Cannot find GPU")
else:
    gpu = torch.device("cuda:0")

# Create the pt model
# n_heads doesn't impact learnable params??
# https://discuss.pytorch.org/t/what-does-increasing-number-of-heads-do-in-the-multi-head-attention/101294/4
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", layer_norm_eps=1e-6, batch_first=True)
trans_nn = torch.nn.TransformerEncoder(encoder_layer, n_layers)
model = ActionTransformer(trans_nn, d_model, num_frames, num_classes, skel_extractor, mlp_head_sz).to(gpu)

# Dummy input which have the same shape as the expected single vector output from openpose (1, 30, 52)
# Ensures we have set up the starting layer(s) of the AcT correctly
inputs = torch.ones(1, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                    config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']).to(gpu)
model(inputs)

# Count number of trainable parameters
# print(count_parameters_pt(model))

# Load keras weights OR check name of weights and their shape
weights_dict = open_pickle("keras_weight_dict.pickle")
for name, param in model.named_parameters():
    if name in weights_dict.keys():
        param.data = torch.tensor(weights_dict[name]).to(gpu)
    else:
        raise Exception(f"weight for {name} is not found ")
    # print(name, param.data.shape)

# Load dummy input and save output as a numpy array
np_input = np.load("test_array.npy")
t_input = torch.tensor(np_input).to(torch.float).to(gpu)
model.eval()
t_output = model(t_input)
# print(t_output)
np.save("pt_output.npy", t_output.cpu().detach().numpy())

