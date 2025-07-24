import torch
from collections import defaultdict

class RolloutBuffer:
    def __init__(self, size, obs_dim, action_dim, device):
        self.size = size
        self.device = device
        self.ptr = 0
        self.full = False
        self.data = defaultdict(list)

    def add(self, **kwargs):
        for k, v in kwargs.items():
            self.data[k].append(v)
        self.ptr += 1
        if self.ptr >= self.size:
            self.full = True

    def get(self):
        batch = {k: torch.as_tensor(torch.stack(v), device=self.device) for k, v in self.data.items()}
        self.ptr = 0
        self.data.clear()
        self.full = False
        return batch