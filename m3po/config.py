from dataclasses import dataclass, asdict

@dataclass
class Config:
    # --- env ---
    num_tasks: int = 50
    num_envs: int = 50  # parallel envs
    max_episode_steps: int = 500

    # --- model ---
    obs_dim: int = 39  # Sawyer state + object/goal (MT50 default)
    action_dim: int = 4  # xyz + gripper open/close
    latent_dim: int = 64
    embed_dim: int = 8

    # --- planning ---
    horizon: int = 5
    num_candidates: int = 64
    plan_iters: int = 1
    mppi_lambda: float = 1.0
    mppi_noise: float = 0.3

    # --- algo ---
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ppo_clip: float = 0.2
    entropy_coef: float = 1e-3
    bonus_beta_start: float = 1.0
    bonus_beta_end: float = 0.0
    bonus_decay_steps: int = 1_000_000

    # lr
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_model: float = 1e-3

    steps_per_iter: int = 10240
    ppo_epochs: int = 3
    minibatch_size: int = 256

    device: str = "cuda"

    def to_dict(self):
        return asdict(self)

def default_cfg() -> Config:
    return Config()