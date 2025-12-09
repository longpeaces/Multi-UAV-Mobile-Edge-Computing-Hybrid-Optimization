# Multi-UAV Assisted Mobile Edge Computing: A Hybrid Optimization Approach

## Objective

The primary objective of this research is to develop a framework for a multi-UAV-assisted collaborative Mobile Edge Computing (MEC) network. We aim to jointly optimize four interdependent components: task offloading decisions, service caching placement, content caching strategies, and UAV trajectories. The goal is to minimize a weighted sum of service latency and system-wide energy consumption while simultaneously maximizing user fairness.

We are aiming to implement a hybrid optimization approach that combines **multi-agent deep reinforcement learning** with **collaborative and adaptive caching policies**. We are trying to create a generic framework that can be used with different models for finding the best-suited one for our purpose. Also trying to incorporate modern Python practices and type annotations. Developed using Python 3.12.0 and PyTorch 2.8.0.

Currently included MARL models:
- MADDPG (Multi-Agent Deep Deterministic Policy Gradient)
- MATD3 (Multi-Agent Twin Delayed Deep Deterministic Policy Gradient)
- MAPPO (Multi-Agent Proximal Policy Optimization)
- MASAC (Multi-Agent Soft Actor-Critic)
- Random baseline

![System Model](docs/system_model.jpg)

Directory Structure:

```
.
├── environment
│   ├── comm_model.py
│   ├── env.py
│   ├── uavs.py
│   └── user_equipments.py
├── marl_models
│   ├── base_model.py
│   ├── buffer_and_helpers.py
│   ├── utils.py
│   ├── maddpg
│   │   ├── agents.py
│   │   └── maddpg.py
│   ├── matd3
│   │   ├── agents.py
│   │   └── matd3.py
│   ├── mappo
│   │   ├── agents.py
│   │   └── mappo.py
│   ├── masac
│   │   ├── agents.py
│   │   └── masac.py
│   └── random_baseline
│       └── random_model.py
├── utils
│   ├── logger.py
│   ├── plot_logs.py
│   └── plot_snapshots.py
├── config.py
├── train.py
├── test.py
├── main.py
├── visualize.py
├── requirements.txt
├── README.md
├── .gitignore
└── LICENSE
```

Packages currently required to be installed (can be found in `requirements.txt`):
- torch
- numpy
- matplotlib

## Usage Instructions

### Quickstart

```bash
# 1) Clone this repo locally
git clone git@github.com:Project-Group-BTP/Multi-UAV-Mobile-Edge-Computing-Hybrid-Optimization.git
cd Multi-UAV-Mobile-Edge-Computing-Hybrid-Optimization

# 2) Create and activate a Python 3.12+ virtual environment (example using venv)
python -m venv .venv
source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt

# 4) Run a short MATD3 smoke test with positive reward shaping
python main.py train --num_episodes=1 --model=matd3 --steps_per_episode=50 --initial_random_steps=10

# 5) Compare against the random baseline under the same settings
python main.py train --num_episodes=1 --model=random --steps_per_episode=50

# 6) Plot the rewards from any generated log files (works without matplotlib)
python utils/svg_plot.py --logs sample_logs/sample_train_logs/log_data_*.json \
    --metric reward --output sample_logs/sample_train_plots/quick_reward_compare.svg
```

These commands keep the convergence objective positive by default (see the "Keeping the reward positive" section below). Increase `num_episodes` and `steps_per_episode` for full training runs. Logs are saved as `log_data_*.json` inside the model's output directory and can be plotted with `utils/svg_plot.py` or any JSON-friendly plotting tool.

### Customizing and extending

The code has been designed to be modular and extensible. You can easily add new models along the lines of existing models by creating a new folder in `marl_models` with its algorithm and agents. Update the `main.py` file and `get_model` function in `marl_models/utils.py` to include your new model.

All settings and hyperparameters are configurable in `config.py` (including reward targets for positive returns). Refer to the inline docs/comments in that file for details.

### Saving to your own GitHub repository

If you want this code in a personal GitHub repo, you can push the current tree to a new remote:

```bash
# From inside the cloned project
git remote remove origin                # optional: detach from the template origin
git remote add origin <your_repo_url>   # e.g., git@github.com:<you>/<repo>.git
git push -u origin main                 # or "master" depending on your default branch
```

If you prefer to keep both remotes, simply add a second remote (e.g., `git remote add mine <your_repo_url>`) and push to that instead of replacing `origin`.

### Training

It can be used to start training from scratch or resume training from a previously saved checkpoint.

```bash
# Start training from scratch
python main.py train --num_episodes=<total_episodes>

# Optional overrides for quick smoke tests without editing config.py
python main.py train --num_episodes=1 --model=matd3 --steps_per_episode=50 --initial_random_steps=10

# To resume training from a saved checkpoint, specify no. of additional episodes, path to the saved checkpoint, and path to the saved config file (to load and use the same settings).
python main.py train --num_episodes=<additional_episodes> --resume_path="<path_to_checkpoint_directory>" --config_path="<path_to_saved_config>"
```

### Testing

It can be used to test a saved model for a specified number of episodes.
To test a saved model, you must provide the path to the model's directory and its corresponding configuration file.

```bash
# Start testing, with saved model path and config file saved during that model's training run (to load and use the same settings).
python main.py test --num_episodes=<total_episodes> --model_path="<path_to_model_directory>" --config_path="<path_to_saved_config>"

# You can also override the model or episode length for quick baseline comparisons
python main.py test --num_episodes=1 --model=random --steps_per_episode=50 --model_path="<path_to_model_directory>" --config_path="<path_to_saved_config>"
```

### Keeping the reward positive and aligned with the paper objective

The reward now compares latency, energy cost, and fairness against tunable targets so that well-performing policies yield positi
ve values. Adjust these in `config.py` as needed:

```python
FAIRNESS_TARGET = 0.8        # desired Jain's fairness index (0-1)
LATENCY_TARGET = 1e5         # target aggregated latency; lower latency produces positive terms
ENERGY_COST_TARGET = 1.0     # target combined grid+market energy cost (Wh-equivalent)
REWARD_OFFSET = 0.0          # increase to uniformly shift rewards upward if you want strictly positive returns
```

If your workloads have higher typical latency or energy costs, raise the corresponding targets so the normalized reward remai
ns positive after convergence. You can validate the shaping quickly with short runs, e.g.:

```bash
python main.py train --num_episodes=1 --model=matd3 --steps_per_episode=50 --initial_random_steps=10
```

A temporary script to run the environment with random actions and visualize the state for just testing the environment alone. Run using:

```bash
python visualize.py
```

### Lightweight log plotting (no external libraries)

If you cannot install matplotlib in a restricted environment, you can still turn existing `log_data_*.json` files into SVG comparison plots.

```bash
# Generate a reward comparison SVG from one or more training logs
python utils/svg_plot.py --logs sample_logs/sample_train_logs/log_data_2025-10-20_10-50-12.json \
    sample_logs/sample_train_logs/log_data_2025-10-20_10-42-58.json \
    --metric reward --output sample_logs/sample_train_plots/offline_reward_compare.svg
```

The script auto-detects whether the log is indexed by episodes or updates and will plot all provided runs on the same axes for quick visual benchmarking.

**PS: Currently under rapid development and may be subject to significant changes.**

## Contributors

- Roopam Taneja
- Vraj Tamakuwala

### Made with ❤️
