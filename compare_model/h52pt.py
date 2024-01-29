"""
Code to convert keras weights to pytorch weights using the custom transformer
"""
import sys

import numpy as np
Q_WEIGHTS = [0, 1]
K_WEIGHTS = [2, 3]
V_WEIGHTS = [4, 5]
OUT_WEIGHTS = [6, 7]
LIN1_WEIGHTS = [8, 9]
LIN2_WEIGHTS = [10, 11]
NORM1_WEIGHTS = [12, 13]
NORM2_WEIGHTS = [14, 15]


def change_dict(dict_pt, k_name, w_name_pt, w_value):
    """
    Modify dictionary storing pt weights
    Args:
        dict_pt: dict | dictionary to modify that stores pt weights
        k_name: weight name in keras
        w_name_pt: string | name of the corresponding pt weight
        w_value: np array | weight value from keras unchanged

    Returns: None
    """
    if "kernel" in k_name:
        dict_pt[w_name_pt + ".weight"] = np.swapaxes(w_value, 0, 1)
    elif "bias" in k_name:
        dict_pt[w_name_pt + ".bias"] = w_value
    elif "gamma" in k_name:
        dict_pt[w_name_pt + ".weight"] = w_value
    elif "beta" in k_name:
        dict_pt[w_name_pt + ".bias"] = w_value
    else:
        raise Exception(f"Undefined weight type: {k_name}?")


def weight_x(num_t_layer, dict_pt, w_info, transformer_layer=False):
    """
    Saves the weights from keras to pytorch
    Args:
        num_t_layer: int | number of layers in the transformer encoder
        dict_pt: dict | to save weights in
        w_info: weight information from keras model
        transformer_layer: bool | if true, w_info will be a list of weights

    Returns: None
    """

    if transformer_layer:  # In this case, w_info is a list of the weights for each encoder layer
        t_layer = w_info[0].name.split('/')[2]  # Obtain which encoder layer we are up to
        if t_layer[-1].isnumeric():  # First layer has no numeric value
            numeric = t_layer[-1]
        else:
            numeric = 0
        pt_layer_name = f"transformer.encoder_layers.{numeric}."

        # iterate through the different layers of the transformer encoder
        for w_idx, w_transf in enumerate(w_info):
            if w_idx in Q_WEIGHTS:  # Query
                change_dict(dict_pt, w_transf.name, pt_layer_name + "mha.wq", w_transf.numpy())
            elif w_idx in K_WEIGHTS:  # Key
                change_dict(dict_pt, w_transf.name, pt_layer_name + "mha.wk", w_transf.numpy())
            elif w_idx in V_WEIGHTS:  # Value
                change_dict(dict_pt, w_transf.name, pt_layer_name + "mha.wv", w_transf.numpy())
            elif w_idx in OUT_WEIGHTS:  # Linear layer in mha
                change_dict(dict_pt, w_transf.name, pt_layer_name + "mha.dense", w_transf.numpy())
            elif w_idx in LIN1_WEIGHTS:  # Linear layer pre-activation
                change_dict(dict_pt, w_transf.name, pt_layer_name + "ffn1", w_transf.numpy())
            elif w_idx in LIN2_WEIGHTS:  # Liner layer post-activation
                change_dict(dict_pt, w_transf.name, pt_layer_name + "ffn2", w_transf.numpy())
            elif w_idx in NORM1_WEIGHTS:  # First norm layer
                change_dict(dict_pt, w_transf.name, pt_layer_name + "layernorm1", w_transf.numpy())
            elif w_idx in NORM2_WEIGHTS:  # Second norm layer
                change_dict(dict_pt, w_transf.name, pt_layer_name + "layernorm2", w_transf.numpy())
    elif "dense" in w_info.name:
        dense_proj = num_t_layer*6  # 6 dense layer per transformer layer starting at count 0
        # Dense layers outside the transformer architecture
        if f"dense_{dense_proj}/" in w_info.name:
            change_dict(dict_pt, w_info.name, "project_higher", w_info.numpy())
        elif f"dense_{dense_proj+1}/" in w_info.name:
            change_dict(dict_pt, w_info.name, "fc1", w_info.numpy())
        elif f"dense_{dense_proj+2}/" in w_info.name:
            change_dict(dict_pt, w_info.name, "fc2", w_info.numpy())
    elif "class_token" in w_info.name:
        dict_pt["class_token"] = w_info.numpy()
    elif "patch_class" in w_info.name:
        dict_pt["position_embedding.weight"] = w_info.numpy()
    else:
        print("########## error ##########")
        print(w_info.name)
        print("########## error ##########")
        raise Exception("Unidentified weight name. See above")

