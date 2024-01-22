import sys
import os
import h52pt
import torch.nn

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Stops NUMA error
import numpy as np
import pickle
from utils.data import load_mpose
from utils.tools import read_yaml
from utils.transformer import TransformerEncoder, PatchClassEmbedding
import tensorflow as tf
from keras.utils.layer_utils import count_params
from pickle_wrapper import *
# Constants
config = read_yaml("utils/config.yaml")
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

# Number of different layers in a transformer encoder block
NUM_DENSE = 6
NUM_NORM = 2
T_WPL = 16

def build_act(transformer):
    """
    Runs a dummy input through a chain of forward passes to obtain an output. This simulates the series of forward
    passes the network should run.
    """
    # print(f"d_model: {d_model}\n##########")

    # Dummy input which contains the expected shape from OpenPose. (batch_sz, num_frame, num_keypoints)
    inputs = tf.keras.layers.Input(shape=(config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                                          config[config['DATASET']]['KEYPOINTS'] * config['CHANNELS']))
    # print(f"output from OpenPose {inputs.shape}")

    # Embedding step which projects the input to a higher dimension. In this case, the num_keypoints --> d_model
    x = tf.keras.layers.Dense(d_model)(inputs)
    # print(f"input projected into higher dimension {x.shape}")

    # Concatenates a cls token to the input vector. This resulting vector is then summed with a learnable vector
    x = PatchClassEmbedding(d_model, config[config['DATASET']]['FRAMES'] // config['SUBSAMPLE'],
                            pos_emb=None)(x)
    # print(f"Transformer input, vector which has been summed with learnable vector and a cls token added {x.shape}.")
    # TODO: Why is the cls added on top of the columns?? Is the sequence based on the time (ie frames) or the columns?
    # Is it a classification of how the specific joint moved over time?

    # Pass it through the transformer encoders
    x = transformer(x)
    # print(f"transformer o/p {x.shape}")

    # Obtain the cls tokens.
    x = tf.keras.layers.Lambda(lambda x: x[:, 0, :])(x)
    # print(f"obtain all the cls tokens for each sequence {x.shape}")

    # Pass it through the final feedforward network w/ two layers.
    # The first expands the dimension to 4*d_model then the second, which normally reduces it back to normal, is the
    # class prediction
    x = tf.keras.layers.Dense(mlp_head_size)(x)
    # print(f"o/p of first MLP {x.shape}")
    outputs = tf.keras.layers.Dense(config[config['DATASET']]['CLASSES'])(x)
    # print(f"class predicitons: {outputs.shape}")

    # Create the model https://www.tensorflow.org/api_docs/python/tf/keras/Model
    return tf.keras.models.Model(inputs, outputs)


# Use GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[config['GPU']], True)
tf.config.experimental.set_visible_devices(gpus[config['GPU']], 'GPU')

# Create the keras model and load weights
trans = TransformerEncoder(d_model, n_heads, d_ff, dropout, activation, n_layers)
model = build_act(trans)
model.load_weights(config['WEIGHTS'])

# Count trainable params
# trainable_count = count_params(model.trainable_weights)
# print(trainable_count) 227156
# print(model.summary())

# Load dummy input and save output as a numpy array
# np_input = np.load("test_np_array.npy")
# t_input = tf.convert_to_tensor(np_input)
# t_output = model(t_input)
# np.save("tf_output.npy", t_output.numpy())

print("########## SAVE WEIGHTS IN DICT ##########")
weight_dict = {}
count_dense = 0
count_mha = 0
count_norm = 0
for layer in model.layers:
    if (not layer.trainable) or ('input' in layer.name):
        continue

    # print(f"{layer.name} has input shape: {layer.output_shape} and output shape: {layer.output_shape}")
    # print("The weight names and shapes of this layer are as follows")
    if "transformer_encoder" in layer.name:
        for i in range(0, n_layers*T_WPL, 16):
            tl_weights = layer.weights[i:i+T_WPL]  # Weights of each layer
            h52pt.weight_x(count_mha, weight_dict, tl_weights, True)
    else:

        for w in layer.weights:
            h52pt.weight_x(count_mha, weight_dict, w)
            # print(f"{w.name} {w.shape}")
            # print(np.swapaxes(w.numpy(), 0, 1).shape)


save_pickle("keras_weight_dict", weight_dict)
# for w in weight_dict.keys():
#     print(w, weight_dict[w].shape)


