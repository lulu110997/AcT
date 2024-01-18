import sys

import numpy as np
Q_WEIGHTS = [0, 1]
KV_WEIGHTS = list(range(2, 6))
OUT_WEIGHTS = [6, 7]
LIN1_WEIGHTS = [8, 9]
LIN2_WEIGHTS = [10, 22]
NORM1_WEIGHTS = [12, 13]
NORM2_WEIGHTS = [14, 15]


def change_dict(dict_pt, w_type, w_name_pt, w_value):
    """
    Modify dictionary storing pt weights
    Args:
        dict_pt: dict | dictionary to modify that stores pt weights
        w_type: type of weight (eg bias, gamma etc)
        w_name_pt: string | name of the corresponding pt weight
        w_value: np array | weight value from keras unchanged

    Returns: None
    """
    if "kv_kernel" == w_type:
        top = dict_pt[w_name_pt]
        bot = np.swapaxes(w_value.numpy(), 0, 1)
        dict_pt[w_name_pt] = np.vstack((top, bot))
    elif "kv_bias" == w_type:
        top = dict_pt[w_name_pt]
        bot = w_value.numpy()
        dict_pt[w_name_pt] = np.vstack((top, bot))
    elif "kernel" == w_type:
        dict_pt[w_name_pt] = np.swapaxes(w_value, 0, 1)
    elif "bias" == w_type:
        dict_pt[w_name_pt] = w_value.numpy()
    elif "gamma" == w_type:
        dict_pt[w_name_pt] = w_value.numpy()
    elif "beta" == w_type:
        dict_pt[w_name_pt] = w_value.numpy()
    else:
        raise Exception(f"Undefined weight type: {w_type}?")


def weight_x(count, dict_pt, w_info, transformer_layer=False):
    """
    Saves the weights from keras to pytorch
    Args:
        count: int | layer count
        dict_pt: dict | to save weights in
        w_info: weight information from keras model
        transformer_layer: bool | if the

    Returns: int | count incremented by 1
    """
    if "transformer_encoder" in w_info.name:
        t_layer = w_info.weights[0].name.split('/')[2]  # Obtain which encoder layer we are up to
        print(t_layer)
        print(t_layer[-1])
        if t_layer[-1].isnumeric():  # First layer has no numeric value
            numeric = t_layer[-1]
        else:
            numeric = 0
        pt_layer_name = f"transformer.layers.{numeric}."

        # iterate through the different layers of the transformer encoder
        for w_idx, w_transf in enumerate(w_info.weights):
            if w_idx in Q_WEIGHTS:  # Query
                if "kernel" in w_transf.name:
                    dict_pt[pt_layer_name + "attn.in_proj_weight"] = np.swapaxes(w_transf.numpy(), 0, 1)
                elif "bias" in w_transf.name:
                    dict_pt[pt_layer_name + "attn.in_proj_bias"] = w_transf.numpy()
                else:
                    print(w_transf.name)
                    raise Exception("not kernel or bias?")
            elif w_idx in KV_WEIGHTS:  # Key and value
                if "kernel" in w_transf.name:
                    top = dict_pt[pt_layer_name + "attn.in_proj_weight"]
                    bot = np.swapaxes(w_transf.numpy(), 0, 1)
                    dict_pt[pt_layer_name + "attn.in_proj_weight"] = np.vstack((top, bot))
                elif "bias" in w_transf.name:
                    top = dict_pt[pt_layer_name + "attn.in_proj_bias"]
                    bot = w_transf.numpy()
                    dict_pt[pt_layer_name + "attn.in_proj_bias"] = np.vstack((top, bot))
            elif w_idx in LIN1_WEIGHTS:  # Linear layer for reduction
                if "kernel" in w_transf.name:
                    dict_pt[pt_layer_name + "linear1.weight"] = w_transf.numpy()
                elif "bias" in w_transf.name:
                    dict_pt[pt_layer_name + "linear1.bias"] = w_transf.numpy()
                else:
                    print(w_transf.name)
                    raise Exception("not kernel or bias?")
            elif w_idx in LIN2_WEIGHTS:  # Liner layer after reduction
                if "kernel" in w_transf.name:
                    dict_pt[pt_layer_name + "linear2.weight"] = w_transf.numpy()
                elif "bias" in w_transf.name:
                    dict_pt[pt_layer_name + "linear2.bias"] = w_transf.numpy()
                else:
                    print(w_transf.name)
                    raise Exception("not kernel or bias?")
            elif w_idx in NORM1_WEIGHTS:  # First norm layer
                if "gamma" in w_transf.name:
                    dict_pt[pt_layer_name + "norm1.weight"] = w_transf.numpy()
                elif "beta" in w_transf.name:
                    dict_pt[pt_layer_name + "norm1.bias"] = w_transf.numpy()
                else:
                    print(w_transf.name)
                    raise Exception("not gamma or beta?")
            elif w_idx in NORM2_WEIGHTS:  # Second norm layer
                if "gamma" in w_transf.name:
                    dict_pt[pt_layer_name + "norm2.weight"] = w_transf.numpy()
                elif "beta" in w_transf.name:
                    dict_pt[pt_layer_name + "norm2.bias"] = w_transf.numpy()
                else:
                    print(w_transf.name)
                    raise Exception("not gamma or beta?")
    elif "dense" in w_info.name:
        # Dense layers outside the transformer architecture
        if f"dense_24/kernel:0" in w_info.name:
            dict_pt["project_higher.weight"] = np.swapaxes(w_info.numpy(), 0, 1)
        elif f"dense_24/bias:0" in w_info.name:
            dict_pt["project_higher.bias"] = w_info.numpy()
        elif f"dense_25/kernel:0" in w_info.name:
            dict_pt["fc1.weight"] = np.swapaxes(w_info.numpy(), 0, 1)
        elif f"dense_25/bias:0" in w_info.name:
            dict_pt["fc1.bias"] = w_info.numpy()
        elif f"dense_26/kernel:0" in w_info.name:
            dict_pt["fc2.weight"] = np.swapaxes(w_info.numpy(), 0, 1)
        elif f"dense_26/bias:0" in w_info.name:
            dict_pt["fc2.bias"] = w_info.numpy()
    elif "class_token" in w_info.name:
        dict_pt["class_token"] = w_info.numpy()
    elif "patch_class" in w_info.name:
        dict_pt["pos_embedding"] = w_info.numpy()
    else:
        print("########## error ##########")
        print(w_info.name)
        print("########## error ##########")
        raise Exception("Unidentified weight name. See above")

    count += 1
    return count
