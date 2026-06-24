# 🏎️ RL Benchmark Lab — CarRacing-v3

One-click benchmark framework comparing **DQN**, **Double-DQN**, and **PPO** on OpenAI Gymnasium's CarRacing-v3 environment. Built with PyTorch.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/camcimahir/car_racing_gym_DQN_RL/blob/main/notebooks/run_benchmark.ipynb)

> **Try it now** — click the badge above to run the benchmark in Google Colab with zero setup.

---

## Project Structure

```
├── docs/                     # GitHub Pages portfolio site
│   ├── index.html
│   ├── styles.css
│   ├── script.js
│   └── assets/
├── notebooks/
│   └── run_benchmark.ipynb   # One-click Colab notebook
├── task 6/                   # PPO — Actor-Critic, continuous actions, GAE
├── task 7/                   # DQN — Paper-close Double-DQN, RMSProp, Huber
├── task 8/                   # DQN — 48x48 CNN, Adam, MSE
├── task 9/                   # Unified benchmark framework
│   └── rl_benchmark.py
├── Task 6.pdf                # Lab reports
├── Task 7 (1).pdf
├── Task 8.pdf
└── Task 9.pdf
```

## Algorithms

| Algorithm | Architecture | Optimizer | Loss | Key Feature |
|-----------|-------------|-----------|------|-------------|
| **Random** | — | — | — | Sanity-check baseline |
| **DQN (Task 8)** | 48×48 CNN | Adam | MSE | Single-DQN target |
| **Double-DQN (Task 7)** | 84×84 CNN | RMSProp | Huber | Double-DQN action selection |
| **PPO (Task 6)** | 96×96 CNN | Adam | Clipped surrogate | GAE + entropy regularization |

## Quick Start

```bash
# Install dependencies
pip install torch gymnasium[box2d] matplotlib opencv-python

# Run the benchmark (smoke mode — ~3-5 min per algorithm)
cd "task 9"
python rl_benchmark.py --mode smoke

# Other modes
python rl_benchmark.py --mode medium     # ~30 min per algo
python rl_benchmark.py --mode full       # overnight
python rl_benchmark.py --algos dqn ppo   # subset only
```

## Outputs

Each run creates a timestamped folder under `results/` containing:

- `leaderboard.csv` — sorted summary of final performance
- `reward_curves.png` — per-episode + rolling-100 reward
- `loss_curves.png` — training loss over time
- `policy_distribution.png` — action-frequency histogram
- `ppo_components.png` — PPO actor/critic/entropy breakdowns
- `run_log.json` — raw per-episode metrics
- `config.json` — full reproducibility config

## Portfolio Site

The `docs/` folder contains a static site designed for GitHub Pages. To enable it:

1. Go to your repo **Settings → Pages**
2. Set source to **Deploy from a branch**
3. Set branch to `main` and folder to `/docs`
4. Save — your site will be live at `https://camcimahir.github.io/car_racing_gym_DQN_RL/`
