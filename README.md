# M3PO (Massively Multi‑Task Model-Based Policy Optimization) – PyTorch Implementation

This document explains how to **install** and **run** the M3PO codebase It also lists all required dependencies (PyTorch, Meta-World, Gymnasium, TensorBoard, etc.) and optional system packages.

*Note: This repository is supposed to be used for reference purposes only. It is not the official implementation of M3PO*
---

## 1. Quick Start (TL;DR)

```bash
# 1) Clone your repo
git clone https://github.com/Ozzey/M3PO.git m3po
cd m3po

# 2) Create and activate env (conda example)
conda create -n m3po python=3.10 -y
conda activate m3po

# 3) Install Python deps
pip install -r requirements.txt

# 4) (Linux) Export Mujoco key path if needed / set MUJOCO_GL for headless
export MUJOCO_GL=osmesa   # or egl

# 5) Train on Meta-World MT50
python -m m3po.train --env mt50 --config configs/mt50.yaml

# 6) Monitor training
tensorboard --logdir runs

# 7) Evaluate
python -m m3po.evaluate --checkpoint checkpoints/mt50/latest.pt --episodes 10
```

---

## 2. Repository Structure

Your canvas code is provided as a single file with "`# === path ===`" delimiters. Split it into a package layout like:

```
 m3po/
   __init__.py
   config.py
   model.py
   policy.py
   planner.py
   storage.py
   utils.py
   train.py
   evaluate.py
   envs/
     __init__.py
     metaworld_vec_env.py
 configs/
   mt50.yaml
 requirements.txt
 README.md   <- (this file)
```

Feel free to rename or reorganize, but keep imports consistent.

---

## 3. Dependencies

### 3.1 Core Python Libraries

* **Python** ≥ 3.9 (3.10 recommended)
* **PyTorch** ≥ 2.0 (with CUDA if you plan to train on GPU)
* **Gymnasium** ≥ 0.29
* **Meta-World** (latest master or a tagged release supporting Gymnasium)
* **TensorBoard** (for logging)
* **tqdm**, **numpy**, **scipy**, **pyyaml**, **opencv-python** (optional for video), **matplotlib** (optional for plots)

Example `requirements.txt`:

```text
# Core DL
torch>=2.0

# RL / envs
gymnasium>=0.29
metaworld @ git+https://github.com/rlworkgroup/metaworld.git@master
mujoco>=2.3.0  # official mujoco python bindings
# If your Meta-World fork still needs mujoco-py, add instead:
#mujoco-py==2.1.2.14

# Utilities
numpy
scipy
pyyaml
tqdm
opencv-python
matplotlib

# Logging
tensorboard
```

> **Note:** `metaworld` is not always up to date on PyPI. Installing from GitHub ensures you get the latest fixes. If you hit build issues with mujoco-py, switch to the new `mujoco` package (and make sure Meta-World version supports it).

### 3.2 System Packages (Linux/Mac)

* **Mujoco runtime**: If using `mujoco-py`, you need the MuJoCo 2.x binaries and license (for old versions). For the newer `mujoco` package, binaries are downloaded automatically.
* **GL / Headless rendering**: For headless servers, set `MUJOCO_GL=osmesa` or `egl`. Ensure `libosmesa6-dev` or equivalent is installed (Ubuntu). Example:

  ```bash
  sudo apt-get update && sudo apt-get install -y libosmesa6-dev patchelf
  export MUJOCO_GL=osmesa
  ```
* **CMake / Build tools**: Some Meta-World forks require compiling low-level code.

Windows is less tested; WSL2 (Ubuntu) is recommended.

---

## 4. Cloning the Code

```bash
git clone https://github.com/Ozzey/M3PO.git m3po
cd m3po
```

If you don’t have a remote yet, initialize one:

```bash
git init
# add files split out from the canvas
# git add . && git commit -m "Initial M3PO commit"
```

---

## 5. Creating a Virtual Environment

### Conda

```bash
conda create -n m3po python=3.10 -y
conda activate m3po
```

### venv (pure Python)

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
```

---

## 6. Installing Python Requirements

```bash
pip install --upgrade pip wheel setuptools
pip install -r requirements.txt
```

If PyTorch with CUDA is desired, install the matching wheel from [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) (or via conda):

```bash
pip install torch==2.2.2+cu121 torchvision==0.17.2+cu121 torchaudio==2.2.2 --index-url https://download.pytorch.org/whl/cu121
```

(Adjust CUDA version according to your system.)

---

## 7. Environment Setup Notes (Meta-World & Mujoco)

* **Meta-World** expects MuJoCo. With `mujoco` (new bindings) you’re set; it downloads binaries to `~/.mujoco/mujoco210/` etc.
* If you use `mujoco-py`, export `LD_LIBRARY_PATH` and copy MuJoCo binaries to `~/.mujoco`. Follow their README.
* For headless servers with no display:

  ```bash
  export MUJOCO_GL=osmesa  # or egl
  ```
* Test that Meta-World works:

  ```python
  import metaworld
  from metaworld.envs.mujoco.multitask_env import MT50
  mt50 = MT50()
  print(len(mt50.train_classes))  # should be 50
  ```

---

## 8. Running Training

### 8.1 Configure

Edit `configs/mt50.yaml` (or a `.py`/`.json` config) to set:

* Horizon (`H`), candidates (`N`), PPO clip, bonus weight, etc.
* Number of parallel envs, total training steps.

Example minimal YAML:

```yaml
seed: 0
num_envs: 50
num_tasks: 50
embed_dim: 8
latent_dim: 64
planner:
  horizon: 5
  candidates: 64
  iterations: 1
ppo:
  clip: 0.2
  epochs: 4
  batch_size: 8192
  lambda: 0.95
  gamma: 0.99
  entropy_coef: 0.0
bonus:
  weight: 1.0
  anneal_to: 0.0
  anneal_steps: 1_000_000
optim:
  actor_lr: 3e-4
  critic_lr: 3e-4
  model_lr: 1e-3
steps_per_iter: 10240
iters: 1000
log_dir: runs
ckpt_dir: checkpoints
```

### 8.2 Launch Training

```bash
python -m m3po.train --env mt50 --config configs/mt50.yaml
```

Common flags you can add:

* `--device cuda` or `--device cpu`
* `--resume checkpoints/mt50/latest.pt`
* `--wandb` if you integrate Weights & Biases logging

### 8.3 TensorBoard

```bash
tensorboard --logdir runs
```

Open `http://localhost:6006` in your browser to see curves (reward, losses, bonus, etc.).

---

## 9. Evaluation

After training, evaluate:

```bash
python -m m3po.evaluate \
  --checkpoint checkpoints/mt50/latest.pt \
  --episodes 10 \
  --env mt50
```

This will:

* Load encoder, dynamics, reward, value, policy, and task embeddings.
* Run MPPI planning at each step (same as training) but without gradients.
* Report average return and success rate per task.

You can disable planning at eval to benchmark pure policy execution by adding a flag like `--no_planner` (if implemented).

---

## 10. Troubleshooting & Tips

| Symptom                                     | Likely Cause                              | Fix                                                          |
| ------------------------------------------- | ----------------------------------------- | ------------------------------------------------------------ |
| `ImportError: No module named 'metaworld'`  | Not installed / wrong venv                | Activate env, re-install Meta-World from GitHub              |
| Mujoco GL errors / "No OpenGL context"      | Headless server without proper MUJOCO\_GL | `export MUJOCO_GL=osmesa` (or `egl`) before running          |
| `ValueError: expected ... shape` in planner | Mismatch in padding/masking across tasks  | Double-check unified obs/action dims and masks               |
| PPO doesn't learn / returns flat            | Bonus too high or value loss dominating   | Tune `bonus.weight`, clip bonus, check normalized advantages |
| GPU OOM during planning                     | Too many candidates/H or envs             | Reduce `candidates`, `horizon`, or batch envs                |

---

## 11. Extending / Modifying

* Swap MPPI with CEM or random shooting by editing `planner.py`.
* Use image observations: replace encoder with CNN.
* Add Dreamer-style symlog reward transforms if needed.
* Integrate replay buffers for the world model (careful: keep policy on-policy).

---

## 12. Citation

If you publish results with this code, please cite the M3PO paper and the Meta-World benchmark:

```
@inproceedings{Narendra2025M3PO,
  title={M3PO: Massively Multi-Task Model-Based Policy Optimization},
  author={Narendra, Aditya and Makarov, Dmitry and Panov, Aleksandr},
  booktitle={IROS},
  year={2025}
}


```

---

**Questions?** Ping me with the error log or part you want to tweak, and I’ll help patch or refactor. Happy training!

