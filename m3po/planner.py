import torch
import torch.nn.functional as F

class MPCPlanner:
    """MPPI/CEM planner (vectorised)."""
    def __init__(self, cfg):
        self.cfg = cfg
        self.h = cfg.horizon
        self.n = cfg.num_candidates
        self.iters = cfg.plan_iters
        self.lmbda = cfg.mppi_lambda
        self.noise_sigma = cfg.mppi_noise

    @torch.no_grad()
    def plan(self, z, task_ids, policy, world_model, value):
        """Plan one action for each latent in batch."""
        device = z.device
        batch = z.shape[0]
        # initial distribution mean from policy
        dist = policy(z, task_ids)
        mean = dist.mean.unsqueeze(1).expand(batch, self.h, -1)  # (B, H, A)
        std = self.noise_sigma

        # Sample action sequences
        actions = mean.unsqueeze(2) + std * torch.randn(batch, self.h, self.n, mean.shape[-1], device=device)
        returns = torch.zeros(batch, self.n, device=device)
        z_repeat = z.unsqueeze(1).repeat(1, self.n, 1)  # (B, N, D)
        task_ids_repeat = task_ids.unsqueeze(1).repeat(1, self.n)
        for t in range(self.h):
            a_t = actions[:, t]  # (B, N, A)
            z_repeat, r_pred = world_model.predict(z_repeat, a_t, task_ids_repeat)
            returns += (self.cfg.gamma ** t) * r_pred
        returns += (self.cfg.gamma ** self.h) * value(z_repeat, task_ids_repeat)
        # MPPI weighting
        weights = F.softmax(returns / self.lmbda, dim=-1)  # (B, N)
        a0 = (weights.unsqueeze(-1) * actions[:, 0]).sum(dim=1)
        return a0