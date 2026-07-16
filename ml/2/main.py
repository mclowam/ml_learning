import torch.nn as nn
import torch.nn.functional as F
import torch
from torchview import draw_graph
n = 5
m = 3

linear_layer = nn.Linear(n, m) #Linear(in_features=5, out_features=3, bias=True)


activation = nn.Tanh()


# x_tensor = torch.tensor([
#     [1,2],
#     [3,4]
# ])


# random_input = torch.randn(5)
#
# z = linear_layer(random_input)
#
# output = activation(z)
#
# FF_layer = nn.Sequential(
#     linear_layer,
#     activation,
# )


 


