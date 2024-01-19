import torch
import torch.nn as nn

class ActionTransformer(nn.Module):
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
            self.in1 = 98  # (x,yz,qx,qy,qz,qw) w/ 14 keypoints
        self.num_classes = num_classes
        self.T = num_frames
        self.d_model = d_model

        # Embedding block which projects the input to a higher dimension. In this case, the num_keypoints --> d_model
        self.project_higher = nn.Linear(self.in1, self.d_model)

        # cls token to concatenate to the projected input
        self.class_token = nn.Parameter(
            torch.randn(1, 1, self.d_model), requires_grad=True
        )  # self.class_embed = self.add_weight(shape=(1, 1, self.d_model),
        # initializer=self.kernel_initializer, name="class_token")

        # Learnable vectors to be added to the projected input
        self.pos_embedding = torch.nn.Parameter(
            torch.randn(self.T + 1, self.d_model), requires_grad=True
        )  # tf.keras.layers.Embedding(input_dim=(self.n_tot_patches), output_dim=self.d_model)

        # Initialise values of cls and pos emb TODO: how is this initialised using keras? What else did they init
        torch.nn.init.normal_(self.class_token, std=0.02)
        torch.nn.init.normal_(self.pos_embedding, std=0.02)

        # Transformer encoder
        self.transformer = transformer

        # Final MLPs
        self.fc1 = nn.Linear(self.d_model, mlp_head_sz)
        self.fc2 = nn.Linear(mlp_head_sz, self.num_classes)

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