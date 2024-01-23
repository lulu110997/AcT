from collections import OrderedDict

import torch
from torch import nn, optim
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from ignite.engine import *
from ignite.handlers import *
from ignite.metrics import *
from ignite.utils import *
from ignite.contrib.metrics.regression import *
from ignite.contrib.metrics import *

# create default evaluator for doctests
def eval_step(engine, batch):
    return batch

default_evaluator = Engine(eval_step)

# create default optimizer for doctests
# param_tensor = torch.zeros([1], requires_grad=True)
# default_optimizer = torch.optim.SGD([param_tensor], lr=0.1)

# create default trainer for doctests
# as handlers could be attached to the trainer,
# each test must define his own trainer using `.. testsetup:`
# def get_default_trainer():
#     def train_step(engine, batch):
#         return batch
#     return Engine(train_step)

# create default model for doctests

default_model = nn.Sequential(OrderedDict([
    ('base', nn.Linear(4, 2)),
    ('fc', nn.Linear(2, 1))
]))

manual_seed(666)
metric = ConfusionMatrix(num_classes=3)
metric.attach(default_evaluator, 'cm')
y_true = torch.tensor([0, 1, 0, 1, 2])
y_pred = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])
state = default_evaluator.run([[y_pred, y_true]]).metrics['cm']

y_true = torch.tensor([0, 1, 0, 1, 2])
y_pred = torch.tensor([
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
    [1.0, 0.0, 0.0],
    [0.0, 1.0, 0.0],
    [0.0, 1.0, 0.0],
])
state += default_evaluator.run([[y_pred, y_true]]).metrics['cm']
cm_np = state.detach().numpy()
ConfusionMatrixDisplay(cm_np).plot()
plt.show()
