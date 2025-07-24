import torch
import torch.nn as nn
from torch.distributions import Normal
from .utils import orthogonal_init

class PolicyPrior(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.embed_dim, 256), nn.ReLU(),
            nn.Linear(256, cfg.action_dim)
        )
        self.log_std = nn.Parameter(torch.zeros(cfg.action_dim))
        self.apply(orthogonal_init)
        self.task_embed = nn.Embedding(cfg.num_tasks, cfg.embed_dim)

    def forward(self, z, task_ids):
        e = self.task_embed(task_ids)
        x = torch.cat([z, e], dim=-1)
        mean = self.fc(x)
        std = self.log_std.exp().expand_as(mean)
        return Normal(mean, std)

    def act(self, z, task_ids, deterministic=False):
        dist = self.forward(z, task_ids)
        action = dist.mean if deterministic else dist.rsample()
        logp = dist.log_prob(action).sum(-1)
        return action, logp