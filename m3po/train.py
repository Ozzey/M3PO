"""Command‑line training entry point."""


import argparse, time
import torch
from m3po import default_cfg, M3POAgent, make_vector_envs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=1000)
    cfg = default_cfg()
    args = parser.parse_args()

    envs = make_vector_envs(cfg.num_envs, cfg.max_episode_steps)
    agent = M3POAgent(cfg)

    obs, _ = envs.reset()
    task_ids = torch.arange(cfg.num_envs) % cfg.num_tasks
    done = torch.zeros(cfg.num_envs)
    for it in range(args.iters):
        for step in range(cfg.steps_per_iter):
            obs_t = torch.tensor(obs, dtype=torch.float32, device=cfg.device)
            task_ids_t = task_ids.to(cfg.device)
            with torch.no_grad():
                actions = agent.act(obs_t, task_ids_t)
                dist = agent.policy(agent.world.encode(obs_t, task_ids_t), task_ids_t)
                logp = dist.log_prob(actions).sum(-1)
                value = agent.value(agent.world.encode(obs_t, task_ids_t), task_ids_t)
            obs_next, reward, term, trunc, info = envs.step(actions.cpu().numpy())
            done = term | trunc
            agent.store_transition(obs=obs_t, task=task_ids_t, action=actions, reward=torch.tensor(reward, device=cfg.device),
                                   next_obs=torch.tensor(obs_next, dtype=torch.float32, device=cfg.device),
                                   done=torch.tensor(done, device=cfg.device, dtype=torch.float32),
                                   logp=logp, value=value)
            obs = obs_next
            # reset envs that are done
            if done.any():
                obs[done], _ = envs.reset_done(done)
        # bootstrap value for last state
        with torch.no_grad():
            last_val = agent.value(agent.world.encode(torch.tensor(obs, dtype=torch.float32, device=cfg.device), task_ids.to(cfg.device)), task_ids.to(cfg.device)).mean()
        batch = agent.get_batch(last_val)
        agent.update(batch)
        print(f"Iter {it:04d} | mean reward {batch['reward'].mean().item():.2f}")

if __name__ == "__main__":
    main()
