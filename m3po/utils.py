import torch

def orthogonal_init(layer, gain=1.0):
    if isinstance(layer, torch.nn.Linear):
        torch.nn.init.orthogonal_(layer.weight, gain)
        torch.nn.init.zeros_(layer.bias)