import numpy as np
import gymnasium as gym
from metaworld.envs.mujoco.multitask_env import MT50
from gymnasium.vector import AsyncVectorEnv

TASK_ID_TO_NAME = {i: name for i, name in enumerate(MT50().train_classes)}


def _make_single_env(task_id: int, max_steps: int):
    def _thunk():
        env = MT50()
        env.set_task(task_id)
        env = gym.wrappers.TimeLimit(env, max_episode_steps=max_steps)
        return env

    return _thunk


def make_vector_envs(num_envs: int, max_steps: int) -> gym.vector.VectorEnv:
    env_fns = [_make_single_env(i % 50, max_steps) for i in range(num_envs)]
    return AsyncVectorEnv(env_fns)