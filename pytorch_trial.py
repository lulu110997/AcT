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


class AcTorch(nn.Module):
    def __init__(self, transformer_pt, de, d_model, num_frames, num_classes):
        super(AcTorch, self).__init__()

        self.num_classes = num_classes
        self.T = num_frames
        self.d_model = d_model
        self.de = de

        # Embedding block which projects the input to a higher dimension. In this case, the num_keypoints --> d_model
        self.project_higher = nn.Linear(52, self.d_model)

        # cls token to concatenate to the projected input
        self.class_token = nn.Parameter(
            torch.randn(1, 1, self.d_model), requires_grad=True
        )  # self.class_embed = self.add_weight(shape=(1, 1, self.d_model),
        # initializer=self.kernel_initializer, name="class_token")

        # Learnable vectors to be added to the projected input
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(1, self.T + 1, self.d_model), requires_grad=True
        )  # tf.keras.layers.Embedding(input_dim=(self.n_tot_patches), output_dim=self.d_model)

        # Initialise values of cls and pos emb
        torch.nn.init.normal_(self.class_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

        # Transformer encoder
        self.transformer = transformer_pt

        # Final MLPs
        self.fc1 = nn.Linear(64, 4*self.d_model)
        self.fc2 = nn.Linear(4*self.d_model, self.num_classes)

    def forward(self, x):
        batch_sz = x.shape[0]
        x = self.project_higher(x)
        x = x.view(batch_sz, self.d_model, -1).permute(0, 2, 1)
        x = torch.cat([self.class_token.expand(batch_sz, -1, -1), x], dim=1)
        x += self.pos_embedding
        x = self.transformer(x)
        x = x[:, 0, :]
        x = self.fc1(x)
        x = self.fc2(x)
        return x


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

# Create the pt model
encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                           activation="gelu", layer_norm_eps=1e-6, batch_first=True)
trans_nn = nn.TransformerEncoder(encoder_layer, n_layers)
inputs = torch.ones(1, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                        config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS'])


my_model = AcTorch(trans_nn, 64, d_model, 30, 20)
my_model(inputs)
