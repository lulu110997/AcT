# Imports
import torch.utils.data
from transformer_model import ActionTransformer
import yaml
import numpy as np
from scheduler import CustomSchedule
# Define constants
CONFIG_PATH = "../utils/config.yaml"
with open(CONFIG_PATH, "r") as file:
    config = yaml.safe_load(file)
model_sz = config["MODEL_SIZE"]
n_heads = config[model_sz]["N_HEADS"]
d_model = config[model_sz]["EMBED_DIM"]  # Size of embedded input. dv = 64, made constant according to paper
dropout = config[model_sz]["DROPOUT"]
n_layers = config[model_sz]['N_LAYERS']
mlp_head_sz = config[model_sz]["MLP"]  # Output size of the ff layer prior the classification layer
num_frames = config["openpose"]["FRAMES"]
num_classes = config["openpose"]["CLASSES"]
d_ff = 4*d_model  # Output size of the first non-linear layer in the transformer encoder
assert d_model == 64*n_heads
skel_extractor = "openpose"

SCHEDULER = config["SCHEDULER"]
N_EPOCHS = config["N_EPOCHS"]
BATCH_SIZE = config["BATCH_SIZE"]
WEIGHT_DECAY = config["WEIGHT_DECAY"]
WARMUP_PERC = config["WARMUP_PERC"]
STEP_PERC = config["STEP_PERC"]
LR_MULT = config["LR_MULT"]
N_FOLD = config["FOLDS"]
N_SPLITS = config["SPLITS"]
DEVICE = torch.device("cuda:0")

# Build network
encoder_layer = torch.nn.TransformerEncoderLayer(d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout,
                                                 activation="gelu", layer_norm_eps=1e-6, batch_first=True)
transformer = torch.nn.TransformerEncoder(encoder_layer, n_layers)
model = ActionTransformer(transformer, d_model, num_frames, num_classes, skel_extractor, mlp_head_sz)

train_len = 0
test_len = 0
train_steps = np.ceil(float(train_len) / config['BATCH_SIZE'])
test_steps = np.ceil(float(test_len) / config['BATCH_SIZE'])

# https://stackoverflow.com/questions/69576720/implementing-custom-learning-rate-scheduler-in-pytorch
# optimizer = tfa.optimizers.AdamW(learning_rate=lr, weight_decay=self.config['WEIGHT_DECAY'])
optimiser = torch.optim.AdamW(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-07,
                              weight_decay=config['WEIGHT_DECAY'])
lr = CustomSchedule(d_model=d_model, optimizer=optimiser,
                    n_warmup_steps=train_steps * config['N_EPOCHS'] * config['WARMUP_PERC'],
                    lr_mul=train_steps * config['N_EPOCHS'] * config['STEP_PERC'])

# loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True, label_smoothing=0.1)
# https://datascience.stackexchange.com/questions/73093/what-does-from-logits-true-do-in-sparsecategoricalcrossentropy-loss-function
# Might need to implement my own loss to copy keras's categorical cross entropy loss
# https://discuss.pytorch.org/t/cross-entropy-with-one-hot-targets/13580/7
# https://discuss.pytorch.org/t/categorical-cross-entropy-loss-function-equivalent-in-pytorch/85165/7
# or not?
# https://discuss.pytorch.org/t/cant-replicate-keras-categoricalcrossentropy-with-pytorch/146747
loss_fn = torch.nn.CrossEntropyLoss(label_smoothing=0.1, reduction="sum")




# Get data and then train
for split in range(1, N_SPLITS + 1):
    # self.logger.save_log(f"----- Start Split {split} ----\n")

    acc_list = []
    bal_acc_list = []

    for fold in range(N_FOLD):
        # https://stackoverflow.com/questions/69576720/implementing-custom-learning-rate-scheduler-in-pytorch
        # Call scheduler at the right time
        # for i, batch in enumerate(dataloader):
        #     sched.zero_grad()
        #     ...
        #     loss.backward()
        #     sched.step_and_update_lr()
        train_data = get_dataloader(train_idx_list, action_label, fold_idx)

# Training function
def train(epochs, train_data, val_data, fold, split):
    """

    Args:
        epochs:
        train_data:
        val_data:
        fold:
        split:

    Returns: tuple | accuracy of train or val data?
    """

    for epoch in range(epochs):
        model.train()
        for batch_x, batch_y in train_data:
            # Change the label tensor so that all actions that is not of interest is one and actions of interest is
            # zero. In the paper, a true positive is when the model classifies a null action.
            batch_x = torch.swapaxes(batch_x, 1, 2).type(
                'torch.FloatTensor')  # even though dtype is already a float https://stackoverflow.com/questions/44717100/pytorch-convert-floattensor-into-doubletensor
            batch_y = torch.where(batch_y == action_label, 0.0, 1.0).unsqueeze(
                1)  # https://stackoverflow.com/questions/57798033/valueerror-target-size-torch-size16-must-be-the-same-as-input-size-torch
            model.zero_grad()  # https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch
            output = model(batch_x.to(device))
            loss = loss_function(output, batch_y.to(device))
            loss.backward()
            optimizer.step()

def eval(epochs, test_data, fold, split):
    """

    Args:
        epochs:
        test_data:
        fold:
        split:

    Returns: tuple | accuracy and balanced accuracy

    """
    pass