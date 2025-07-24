import torch
import torch.nn as nn
from .utils import orthogonal_init

class Encoder(nn.Module):
    def __init__(self, obs_dim, latent_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(obs_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.apply(orthogonal_init)

    def forward(self, obs, embed):
        x = torch.cat([obs, embed], dim=-1)
        return self.fc(x)

class LatentDynamics(nn.Module):
    def __init__(self, latent_dim, action_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, latent_dim)
        )
        self.apply(orthogonal_init)

    def forward(self, z, a, embed):
        x = torch.cat([z, a, embed], dim=-1)
        return self.fc(x)

class RewardHead(nn.Module):
    def __init__(self, latent_dim, action_dim, embed_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(latent_dim + action_dim + embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.apply(orthogonal_init)

    def forward(self, z, a, embed):
        x = torch.cat([z, a, embed], dim=-1)
        return self.fc(x).squeeze(-1)

class WorldModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.encoder = Encoder(cfg.obs_dim, cfg.latent_dim, cfg.embed_dim)
        self.dynamics = LatentDynamics(cfg.latent_dim, cfg.action_dim, cfg.embed_dim)
        self.reward = RewardHead(cfg.latent_dim, cfg.action_dim, cfg.embed_dim)
        self.task_embed = nn.Embedding(cfg.num_tasks, cfg.embed_dim)

    def get_embed(self, task_ids):
        return self.task_embed(task_ids)

    def encode(self, obs, task_ids):
        return self.encoder(obs, self.get_embed(task_ids))

    def predict(self, z, a, task_ids):
        e = self.get_embed(task_ids)
        z_next = self.dynamics(z, a, e)
        r_pred = self.reward(z, a, e)
        return z_next, r_pred

class ValueFunction(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cfg.latent_dim + cfg.embed_dim, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.apply(orthogonal_init)
        self.task_embed = nn.Embedding(cfg.num_tasks, cfg.embed_dim)

    def forward(self, z, task_ids):
        e = self.task_embed(task_ids)
        x = torch.cat([z, e], dim=-1)
        return self.net(x).squeeze(-1)