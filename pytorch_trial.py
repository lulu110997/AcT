from collections import OrderedDict
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
from transformer_model import ActionTransformer


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


# Check GPU
if not torch.cuda.is_available():
    raise Exception("Cannot find GPU")
else:
    gpu = torch.device("cuda:0")

# Create the pt model
# n_heads doesn't impact learnable params??
# https://discuss.pytorch.org/t/what-does-increasing-number-of-heads-do-in-the-multi-head-attention/101294/4
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", layer_norm_eps=1e-6, batch_first=True)
trans_nn = nn.TransformerEncoder(encoder_layer, n_layers)
inputs = torch.ones(1, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                        config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']).to(gpu)


my_model = ActionTransformer(trans_nn, d_model, 30, 20)
my_model.to(gpu)
my_model(inputs)
# print(count_parameters_pt(my_model))
# X_train, y_train, X_test, y_test = load_mpose(config['DATASET'], split, legacy=config['LEGACY'], verbose=False)
# np.save("test_np_array.npy", X_train[:5,:,:])
# t_input = np.load("test_np_array.npy")
# t_input = torch.from_numpy(t_input).to(torch.float32).to(gpu)
# my_model.eval()
# t_output = my_model(t_input)
# np.save("pt_output.npy", t_output.cpu().detach().numpy())
# my_model = nn.Sequential(
#     nn.Linear(20, 30)
# )
weight_dict = OrderedDict()

modules = my_model.modules()
print(my_model.state_dict())
# for l in list(my_model.named_parameters()):
    # print(l[0], ':', l[1].cpu().detach().numpy().shape)
# for m in modules:
#     print(m)
#     break
    # # if layer.get_config()['name'] == 'patch_class_embedding':
    # print(layer.get_config()['name'])
    # for w in layer.weights:
    #     print(f"{w.name} has shape: {w.shape}")