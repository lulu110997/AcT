import torch

x = torch.randn(2, 3, 2)
print(x.size())
# tensor([[ 1.0028, -0.9893,  0.5809],
#         [-0.1669,  0.7299,  0.4942]])
x = torch.transpose(x, 1, 2)
print(x.size())

# tensor([[ 1.0028, -0.1669],
#         [-0.9893,  0.7299],
#         [ 0.5809,  0.4942]])