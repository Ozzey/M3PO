"""Massively Multi‑Task Model‑Based Policy Optimisation (M3PO).

High‑level API:
    - make_agent(cfg) -> (agent, envs)
    - agent.train() / agent.evaluate()

See train.py for CLI.
"""
from .config import default_cfg, Config
from .model import WorldModel, ValueFunction
from .policy import PolicyPrior
from .planner import MPCPlanner
from .agent import M3POAgent
from .envs import make_vector_envs, TASK_ID_TO_NAME

__all__ = [
    "WorldModel",
    "ValueFunction",
    "PolicyPrior",
    "MPCPlanner",
    "M3POAgent",
    "make_vector_envs",
    "TASK_ID_TO_NAME",
    "Config",
    "default_cfg",
]