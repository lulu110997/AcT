import sys
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stops NUMA error
import numpy as np
from utils.data import load_mpose
from utils.transformer import TransformerEncoder, PatchClassEmbedding
from utils.tools import read_yaml
import torch.nn as nn
import torch
import tensorflow as tf
from pytorch_model_summary import summary

config = read_yaml("utils/config.yaml")

# Constants
model_size = config['MODEL_SIZE']
split = 1
fold = 0
bin_path = config['MODEL_DIR']
n_heads = config[model_size]['N_HEADS']
n_layers = config[model_size]['N_LAYERS']
embed_dim = config[model_size]['EMBED_DIM']
dropout = config[model_size]['DROPOUT']
mlp_head_size = config[model_size]['MLP']
activation = tf.nn.gelu
d_model = 64 * n_heads
d_ff = d_model * 4
pos_emb = config['POS_EMB']


def count_parameters_pt(model):
    """
    https://discuss.pytorch.org/t/how-do-i-check-the-number-of-parameters-of-a-model/4325/9
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def build_act(transformer):
    # Instantiate an input tensor
    # inputs = torpytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)        config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS'])
    # inputs should have (1, 30, 52)
    inputs = tf.keras.layers.Input(shape=(config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                                          config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']))

    # Embedding step
    x = tf.keras.layers.Dense(d_model)(inputs)
    # x = nn.Linear(inputs, d_model)

    # Tokenise the embedded input
    x = PatchClassEmbedding(d_model, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                            pos_emb=None)(x)
    x = transformer(x)
    x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    outputs = tf.keras.layers.Dense(config[config['DATASET']]['CLASSES'])(x)
    return tf.keras.models.Model(inputs, outputs)

# Check GPU
if not torch.cuda.is_available():
    raise Exception("Cannot find GPU")

# Create the pt model
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", layer_norm_eps=1e-6, batch_first=True)
trans_nn = nn.TransformerEncoder(encoder_layer, n_layers)
inputs = torch.ones(1, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                        config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS'])
tmp = nn.Linear(inputs.shape[2]+1, d_model+1)
# class_token = torch.nn.Parameter(
#     torch.randn(1, 1, inputs[2]))
# pos_embedding = torch.nn.Parameter(
#     torch.randn(1, 31, inputs[2]))
# torch.nn.init.normal_(class_token, std=0.02)
# torch.nn.init.normal_(pos_embedding, std=0.02)

print(summary(trans_nn, inputs, show_input=False))
print(count_parameters_pt(trans_nn))

print(x.shape)
trans = TransformerEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)
build_act(trans)
