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
            transformer: torch.nn.TransformerEncoder | transformer architecture (encoder only)
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

        # cls token to concatenate to the projected input
        self.class_token = torch.nn.Parameter(torch.randn(1, 1, self.d_model), requires_grad=True)
        # self.class_embed = self.add_weight(shape=(1, 1, self.d_model),
        # initializer=self.kernel_initializer, name="class_token")

        # Learnable vectors to be added to the projected input
        # self.position_embedding = torch.nn.Parameter(
        #     torch.randn(self.T + 1, self.d_model), requires_grad=True
        # )  # tf.keras.layers.Embedding(input_dim=(self.n_tot_patches), output_dim=self.d_model)
        # https://stackoverflow.com/questions/71417255/how-should-the-output-of-my-embedding-layer-look-keras-to-pytorch
        self.position_embedding = torch.nn.Embedding(self.T+1, self.d_model)
        self.positions = torch.arange(start=0, end=self.T+1, step=1).to(device)  # Positional vectors??

        # Transformer encoder
        self.transformer = transformer

        # Final MLPs
        self.fc1 = torch.nn.Linear(self.d_model, mlp_head_sz)
        self.fc2 = torch.nn.Linear(mlp_head_sz, self.num_classes)

        # Initialise weights of layers TODO: how is this initialised using keras?
        # torch.nn.init.normal_(self.class_token, std=(2.0/self.d_model)**0.5)  # HeNormal
        # torch.nn.init.xavier_uniform_(self.fc1.weight.data)  # glorot_uniform
        # torch.nn.init.xavier_uniform_(self.fc2.weight.data)
        # torch.nn.init.xavier_uniform_(self.project_higher.weight.data)
        # for name, params in self.transformer.named_parameters():
        #     try:
        #         if len(params.data.shape) > 1:
        #             torch.nn.init.xavier_uniform_(params.data)
        #         elif ('bias' in name) and (('linear' in name) or ('attn' in name)):
        #             torch.nn.init.zeros_(params.data)
        #     except:
        #         print(name)

    def forward(self, x):
        batch_sz = x.shape[0]
        x = self.project_higher(x)  # Project to higher dim
        x = torch.cat([self.class_token.repeat(batch_sz, 1, 1), x], dim=1)  # Concatenate cls TODO: correct?
        pe = self.position_embedding(self.positions)  # Feed position vectors to embedding layer??
        x += pe  # Add pos emb to input
        x = self.transformer(x)  # Feed through the transformer
        x = x[:, 0, :]  # Obtain the cls vectors
        x = self.fc1(x)  # Feed through a ff network
        x = self.fc2(x)  # Feed through classification layer
        return x