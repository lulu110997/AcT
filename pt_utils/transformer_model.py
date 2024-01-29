import sys

import torch
# Check GPU
if not torch.cuda.is_available():
    import warnings
    warnings.warn("Cannot find GPU")
    device = "cpu"
else:
    device = "cuda:0"

class ActionTransformer(torch.nn.Module):
    """
    Porting https://github.com/PIC4SeR/AcT from Keras to Pytorch
    """
    def __init__(self, transformer, d_model, num_frames, num_classes, skel_extractor, mlp_head_sz):
        """
        Creates the Action Transformer
        Args:
            transformer: TransformerEncoder | transformer architecture (encoder only)
            d_model: int | Input size of the transformers, ie d_embedded (dimension to project original input vector)
            num_frames: int | Number of frames used to classify an action
            num_classes: int | Number of action classes to identify
            skel_extractor: string | openpose, posenet or nuitrack. Determines initial input size for the embedding layer
            mlp_head_sz: int | Output size of the ff layer prior the classification layer
        """

        super(ActionTransformer, self).__init__()
        if skel_extractor == "openpose":
            self.in1 = 52  # (x,y,vx,vy) w/ 13 keypoints
        elif skel_extractor == "posenet":
            self.in1 = 68  # (x,y,vx,vy) w/ 17 keypoints
        elif skel_extractor == "nuitrack":
            self.in1 = 98  # (x,y,z,qx,qy,qz,qw) w/ 14 keypoints

        self.num_classes = num_classes
        self.T = num_frames
        self.d_model = d_model

        # Embedding block which projects the input to a higher dimension. In this case, the num_keypoints --> d_model
        self.project_higher = torch.nn.Linear(self.in1, self.d_model)

        # CLS token and pos embedding
        # https://github.com/MathInf/toroidal/blob/bff09f725627e4629d464008dc7c5f9d6322ebad/toroidal/models.py#L18
        # https://stackoverflow.com/questions/71417255/how-should-the-output-of-my-embedding-layer-look-keras-to-pytorch

        # cls token to concatenate to the projected input
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.d_model), requires_grad=True)

        # Learnable vectors to be added to the projected input
        self.position_embedding = torch.nn.Embedding(self.T+1, self.d_model)
        self.positions = torch.arange(start=0, end=self.T+1, step=1).to(device)  # Positional vectors??

        # Transformer encoder
        self.transformer = transformer

        # Final MLPs
        self.fc1 = torch.nn.Linear(self.d_model, mlp_head_sz)
        self.fc2 = torch.nn.Linear(mlp_head_sz, self.num_classes)

        # Initialise weights of layers
        self._reset_parameters()

    def forward(self, x):
        batch_sz = x.shape[0]
        x = self.project_higher(x)  # Project to higher dim
        x = torch.cat([self.class_token.expand(batch_sz, -1, -1), x], dim=1)  # Concatenate cls TODO: correct? might also need to use torch.repeat
        pe = self.position_embedding(self.positions)  # Feed position vectors to embedding layer??
        x += pe  # Add pos emb to input
        x = self.transformer(x)  # Feed through the transformer
        x = x[:, 0, :]  # Obtain the cls vectors
        x = self.fc1(x)  # Feed through a ff network
        x = self.fc2(x)  # Feed through classification layer
        return x

    def _reset_parameters(self):
        """
        Resets the parameters of the model to match how they are initialised in keras
        """
        torch.nn.init.normal_(self.class_token, std=(2.0/(self.class_token.data.shape[-1])**0.5))  # HeNormal
        torch.nn.init.uniform_(self.position_embedding.weight.data, a=-0.05, b=0.05)  # Random uniform initializer

        for name, params in self.named_parameters():
            # print(name, params.data.shape)

            # class token and embedding layer have been initalised above
            # layer norms are initialised correctly as by default weights are init as ones and bias as zeros
            if ('weight' in name) and (not ('norm' in name)) and (not ('position_embedding' in name)):
                torch.nn.init.xavier_uniform_(params.data)  # glorot_uniform used to initalise weights
            elif ('bias' in name) and (not ('norm' in name)):
                torch.nn.init.zeros_(params.data)  # biases are initialised to zero

class MultiHeadAttention(torch.nn.Module):
    def __init__(self, d_model, num_heads, depth):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model  # Embedded dim
        self.depth = depth  # Equivalent to head_dim = embed_dim // num_heads which is used to split the projected qkv's

        # Feedforward layers for projecting input as qkv values
        self.wq = torch.nn.Linear(d_model, d_model)
        self.wk = torch.nn.Linear(d_model, d_model)
        self.wv = torch.nn.Linear(d_model, d_model)

        # linear layer after concat
        self.dense = torch.nn.Linear(d_model, d_model)

    def split_heads(self, x, batch_size):
        """Split the last dimension into (num_heads, depth).
        Transpose the result such that the shape is (batch_size, num_heads, seq_len, depth)
        """
        x = torch.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        # return tf.transpose(x, perm=[0, 2, 1, 3])
        return torch.transpose(x, dim0=1, dim1=2)

    def forward(self, v, k, q):
        # batch_size = tf.shape(q)[0]
        batch_size = q.size(dim=0)
        assert q.size(dim=2) == self.d_model

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q, batch_size)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k, batch_size)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v, batch_size)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(q, k, v)

        # scaled_attention = tf.transpose(scaled_attention,
        #                                 perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)
        scaled_attention = torch.transpose(scaled_attention, dim0=1, dim1=2)  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = torch.reshape(scaled_attention, (batch_size, -1, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


class TransformerEncoderLayer(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.dropout = dropout

        assert self.d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.mha = MultiHeadAttention(self.d_model, self.num_heads, self.depth).to(device)

        # self.ffn = point_wise_feed_forward_network(self.d_model, self.d_ff, self.activation)
        self.ffn1 = torch.nn.Linear(d_model, d_ff)  # (batch_size, seq_len, dff)
        self.ffn2 = torch.nn.Linear(d_ff, d_model)  # (batch_size, seq_len, d_model)

        # TODO: check d_model is correct input for normalized_shape
        self.layernorm1 = torch.nn.LayerNorm(d_model, eps=1e-6)
        self.layernorm2 = torch.nn.LayerNorm(d_model, eps=1e-6)

        self.dropout1 = torch.nn.Dropout(self.dropout)
        self.dropout2 = torch.nn.Dropout(self.dropout)

    def forward(self, x):
        attn_output, _ = self.mha(x, x, x)  # (batch_size, input_seq_len, d_model)

        attn_output = self.dropout1(attn_output)
        out1 = self.layernorm1(x + attn_output)  # (batch_size, input_seq_len, d_model)

        ffn_output = self.ffn2(torch.nn.functional.gelu(self.ffn1(out1)))  # (batch_size, input_seq_len, d_model)
        ffn_output = self.dropout2(ffn_output)
        out2 = self.layernorm2(out1 + ffn_output)  # (batch_size, input_seq_len, d_model)

        return out2


class TransformerEncoder(torch.nn.Module):
    def __init__(self, d_model, num_heads, d_ff, dropout, n_layers, **kwargs):
        super(TransformerEncoder, self).__init__(**kwargs)
        self.n_layers = n_layers
        # https://discuss.pytorch.org/t/nested-modules-in-pytorch-and-the-parameters-update/70939/5
        self.encoder_layers = torch.nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, d_ff, dropout).to(device)
                               for i in range(n_layers)])


    def forward(self, x):
        for i in range(self.n_layers):
            x = self.encoder_layers[i](x)

        return x

def scaled_dot_product_attention(q, k, v):
    """Calculate the attention weights.
    q, k, v must have matching leading dimensions.
    k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
    Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    Returns:
    output, attention_weights
    """

    # Q @ K_transpose
    matmul_qk = torch.matmul(q, torch.transpose(k, dim0=2, dim1=3))  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = torch.tensor(k.size(dim=3), dtype=torch.int8)  # dimension of key (ie how many key values is used) for one head
    dk_sqrt = torch.sqrt(dk)
    scaled_attention_logits = torch.divide(matmul_qk, dk_sqrt)

    # softmax is normalized on the last axis (seq_len_k) so that the scores add up to 1
    # this is normalising the matrix such that each 'row' adds up to 1
    attention_weights = torch.softmax(scaled_attention_logits, dim=-1)  # (..., seq_len_q, seq_len_k)

    output = torch.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights
