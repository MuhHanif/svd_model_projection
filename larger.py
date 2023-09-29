import torch
from collections import OrderedDict

# create a model with a fixed input size
model = torch.nn.Linear(5, 3)

# create a bigger tensor with the same structure as the input size
bigger_tensor = torch.randn(10, 5)
bigger_tensor_bias = torch.randn(10)

model_copy = model.state_dict()
model_copy["weight"]=bigger_tensor
model_copy["bias"]=bigger_tensor_bias

# load the bigger tensor into the model
model.load_state_dict(model_copy)
print()