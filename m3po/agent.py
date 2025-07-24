import torch
import torch.nn.functional as F
from torch import optim
from .storage import RolloutBuffer

class M3POAgent:
    def __init__(self, cfg):
        self.cfg = cfg
        self.device = torch.device(cfg.device)
        # modules
        from .model import WorldModel, ValueFunction
        from .policy import PolicyPrior
        from .planner import MPCPlanner

        self.world = WorldModel(cfg).to(self.device)
        self.value = ValueFunction(cfg).to(self.device)
        self.policy = PolicyPrior(cfg).to(self.device)
        self.planner = MPCPlanner(cfg)

        # optimizers
        self.opt_model = optim.Adam(self.world.parameters(), lr=cfg.lr_model)
        self.opt_value = optim.Adam(self.value.parameters(), lr=cfg.lr_critic)
        self.opt_policy = optim.Adam(self.policy.parameters(), lr=cfg.lr_actor)

        self.buffer = RolloutBuffer(cfg.steps_per_iter, cfg.obs_dim, cfg.action_dim, self.device)
        self.step_count = 0

    @torch.no_grad()
    def act(self, obs, task_ids):
        z = self.world.encode(obs, task_ids)
        return self.planner.plan(z, task_ids, self.policy, self.world, self.value)

    def compute_bonus(self, batch):
        # model‑free
        with torch.no_grad():
            v_next = self.value(self.world.encode(batch['next_obs'], batch['task']), batch['task'])
        q_mf = batch['reward'] + self.cfg.gamma * v_next * (1.0 - batch['done'])
        # model‑based
        with torch.no_grad():
            z = self.world.encode(batch['obs'], batch['task'])
            z_pred, r_pred = self.world.predict(z, batch['action'], batch['task'])
            q_mb = r_pred + self.cfg.gamma * self.value(z_pred, batch['task'])
        bonus = (q_mf - q_mb).abs()
        return bonus

    def update(self, batch):
        bonus = self.compute_bonus(batch)
        beta = max(self.cfg.bonus_beta_end, self.cfg.bonus_beta_start - self.step_count / self.cfg.bonus_decay_steps)
        bonus = bonus - bonus.mean()
        adv = batch['adv'] + beta * bonus
        # normalize adv
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)

        bsz = adv.size(0)
        idx = torch.randperm(bsz)
        for _ in range(self.cfg.ppo_epochs):
            for start in range(0, bsz, self.cfg.minibatch_size):
                sl = idx[start:start + self.cfg.minibatch_size]
                # critic
                v_pred = self.value(self.world.encode(batch['obs'][sl], batch['task'][sl]), batch['task'][sl])
                v_loss = F.mse_loss(v_pred, batch['ret'][sl])
                self.opt_value.zero_grad(); v_loss.backward(); self.opt_value.step()
                # actor
                dist = self.policy(self.world.encode(batch['obs'][sl], batch['task'][sl]), batch['task'][sl])
                logp = dist.log_prob(batch['action'][sl]).sum(-1)
                ratio = (logp - batch['logp'][sl]).exp()
                surr1 = ratio * adv[sl]
                surr2 = torch.clamp(ratio, 1 - self.cfg.ppo_clip, 1 + self.cfg.ppo_clip) * adv[sl]
                pi_loss = -torch.min(surr1, surr2).mean() - self.cfg.entropy_coef * dist.entropy().mean()
                self.opt_policy.zero_grad(); pi_loss.backward(); self.opt_policy.step()
        # world model update single epoch
        z = self.world.encode(batch['obs'], batch['task'])
        z_next_pred, r_pred = self.world.predict(z, batch['action'], batch['task'])
        z_next = self.world.encode(batch['next_obs'], batch['task'])
        loss_dyn = F.mse_loss(z_next_pred.detach(), z_next) + F.mse_loss(r_pred, batch['reward'])
        self.opt_model.zero_grad(); loss_dyn.backward(); self.opt_model.step()

    def store_transition(self, **kwargs):
        self.buffer.add(**{k: torch.as_tensor(v, device=self.device) for k, v in kwargs.items()})
        self.step_count += 1

    def ready(self):
        return self.buffer.full

    def get_batch(self, last_val):
        data = self.buffer.get()
        # compute returns and advantages (GAE)
        returns = []
        advs = []
        gae = 0
        for t in reversed(range(self.cfg.steps_per_iter)):
            mask = 1.0 - data['done'][t]
            delta = data['reward'][t] + self.cfg.gamma * last_val * mask - data['value'][t]
            gae = delta + self.cfg.gamma * self.cfg.gae_lambda * mask * gae
            advs.insert(0, gae)
            returns.insert(0, gae + data['value'][t])
            last_val = data['value'][t]
        data['adv'] = torch.stack(advs)
        data['ret'] = torch.stack(returns)
        return data