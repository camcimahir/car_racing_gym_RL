"""
RL Testing Framework — Task 9
=============================

One-click benchmark for the RL algorithms developed across tasks 6, 7, and 8 on
the CarRacing-v3 environment. Run from the command line:

    python rl_benchmark.py                       # default: --mode smoke
    python rl_benchmark.py --mode smoke          # ~3-5 min per algo
    python rl_benchmark.py --mode medium         # ~30 min per algo
    python rl_benchmark.py --mode full           # overnight
    python rl_benchmark.py --algos dqn ppo       # subset
    python rl_benchmark.py --mode smoke --no-render-eval

Algorithms benchmarked
----------------------
* random      — uniform-random baseline (sanity floor)
* dqn         — DQN with 48x48 CNN, Adam + MSE, single-DQN target  (Task 8)
* dqn_paper   — DQN with 84x84 CNN, RMSProp + Huber, Double-DQN     (car_racing_DQN.py)
* ppo         — Actor-Critic PPO with continuous actions, GAE       (Task 6)

Outputs
-------
A timestamped folder under results/ containing:
* run_log.json          per-episode metrics for every algorithm
* leaderboard.csv       sorted summary of final performance
* reward_curves.png     per-episode + rolling-100 reward, all algorithms
* loss_curves.png       training loss, all algorithms with a loss
* policy_distribution.png   action-frequency histogram (DQN-family)
* ppo_components.png    PPO actor/critic/entropy losses (only if PPO ran)
* config.json           reproducibility: mode, hyperparameters, seed, git-like fingerprint

Designed so a Nautilus deployment can `git pull && python rl_benchmark.py --mode full`
and walk away.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import random
import sys
import time
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Any, Callable

import numpy as np

# --- Soft imports: the framework still loads on a box without torch / gym, so
# users can read --help and see the design even if dependencies are missing.
try:
    import torch as T
    import torch.nn as nn
    import torch.nn.functional as F
    import torch.optim as optim
    _HAS_TORCH = True
except ImportError:  # pragma: no cover
    _HAS_TORCH = False

try:
    import gymnasium as gym
    _HAS_GYM = True
except ImportError:  # pragma: no cover
    _HAS_GYM = False

try:
    import cv2
    _HAS_CV2 = True
except ImportError:  # pragma: no cover
    _HAS_CV2 = False

try:
    import matplotlib
    matplotlib.use("Agg")  # headless: works on Nautilus / remote boxes
    import matplotlib.pyplot as plt
    _HAS_MPL = True
except ImportError:  # pragma: no cover
    _HAS_MPL = False


# =============================================================================
# Shared environment helpers
# =============================================================================

DISCRETE_ACTIONS = np.array([
    [0.0,  0.0, 0.0],   # 0: do nothing
    [-1.0, 0.0, 0.0],   # 1: steer left
    [1.0,  0.0, 0.0],   # 2: steer right
    [0.0,  1.0, 0.0],   # 3: gas
    [0.0,  0.0, 0.8],   # 4: brake
    [-1.0, 1.0, 0.0],   # 5: gas + left
    [1.0,  1.0, 0.0],   # 6: gas + right
], dtype=np.float32)
N_DISCRETE = len(DISCRETE_ACTIONS)


def preprocess_frame(frame: np.ndarray, size: int) -> np.ndarray:
    """RGB (96,96,3) -> grayscale (size, size) float32 in [0, 1]."""
    if _HAS_CV2:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        if size != gray.shape[0]:
            gray = cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)
    else:
        # Fallback: numpy-only grayscale + nearest-neighbour resize.
        gray = (0.2989 * frame[:, :, 0]
                + 0.5870 * frame[:, :, 1]
                + 0.1140 * frame[:, :, 2])
        if size != gray.shape[0]:
            idx = (np.linspace(0, gray.shape[0] - 1, size)).astype(np.int64)
            gray = gray[idx][:, idx]
    return gray.astype(np.float32) / 255.0


class FrameStack:
    """Stack the last `n` preprocessed frames into a (n, size, size) tensor."""

    def __init__(self, n: int = 4, size: int = 48):
        self.n = n
        self.size = size
        self.frames: collections.deque[np.ndarray] = collections.deque(maxlen=n)

    def reset(self, raw: np.ndarray) -> np.ndarray:
        f = preprocess_frame(raw, self.size)
        self.frames.clear()
        for _ in range(self.n):
            self.frames.append(f)
        return np.stack(self.frames, axis=0)

    def step(self, raw: np.ndarray) -> np.ndarray:
        self.frames.append(preprocess_frame(raw, self.size))
        return np.stack(self.frames, axis=0)


# =============================================================================
# Benchmark config + per-mode budgets
# =============================================================================

@dataclass
class BenchmarkConfig:
    mode: str = "smoke"
    seed: int = 0
    env_name: str = "CarRacing-v3"
    max_episode_steps: int = 1_000        # cap to keep wallclock predictable
    out_dir: str = "results"

    @property
    def episodes(self) -> int:
        return {"smoke": 30, "medium": 200, "full": 1000}[self.mode]

    @property
    def replay_size(self) -> int:
        return {"smoke": 10_000, "medium": 100_000, "full": 300_000}[self.mode]

    @property
    def ppo_total_steps(self) -> int:
        # PPO learns from environment steps, not episodes; scale comparably.
        return {"smoke": 50_000, "medium": 400_000, "full": 2_000_000}[self.mode]

    @property
    def warmup_steps(self) -> int:
        return {"smoke": 500, "medium": 2_000, "full": 20_000}[self.mode]


# =============================================================================
# Common Algorithm interface
# =============================================================================

@dataclass
class EpisodeRecord:
    episode: int
    score: float
    steps: int
    loss: float | None = None
    epsilon: float | None = None
    extra: dict[str, float] = field(default_factory=dict)


class Algorithm:
    """Base class — all benchmarkable algorithms implement this."""
    name: str = "base"
    needs_loss: bool = False
    needs_epsilon: bool = False

    def train(self, env, cfg: BenchmarkConfig) -> list[EpisodeRecord]:
        raise NotImplementedError

    def action_counts(self) -> dict[int, int] | None:
        """Optional: action-frequency histogram for the policy plot."""
        return None


# -----------------------------------------------------------------------------
# Random baseline
# -----------------------------------------------------------------------------
class RandomAgent(Algorithm):
    name = "random"

    def __init__(self):
        self._counts = collections.Counter()

    def train(self, env, cfg):
        records = []
        for ep in range(cfg.episodes):
            obs, _ = env.reset(seed=cfg.seed + ep)
            done = False
            score = 0.0
            steps = 0
            while not done and steps < cfg.max_episode_steps:
                a = np.random.randint(N_DISCRETE)
                self._counts[a] += 1
                obs, r, term, trunc, _ = env.step(DISCRETE_ACTIONS[a])
                score += r
                steps += 1
                done = term or trunc
            records.append(EpisodeRecord(ep, score, steps))
            _log_episode(self.name, ep, cfg.episodes, score,
                         np.mean([r.score for r in records[-100:]]))
        return records

    def action_counts(self):
        return dict(self._counts)


# -----------------------------------------------------------------------------
# DQN (Task 8) — 48x48 CNN, Adam, MSE, single-DQN target
# -----------------------------------------------------------------------------
if _HAS_TORCH:

    class _DQN48(nn.Module):
        def __init__(self, n_actions, in_ch=4):
            super().__init__()
            self.conv1 = nn.Conv2d(in_ch, 32, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
            self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
            self.fc1 = nn.Linear(64 * 2 * 2, 512)
            self.fc2 = nn.Linear(512, n_actions)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = F.relu(self.conv3(x))
            x = x.reshape(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)

    class _DQN84(nn.Module):
        """Paper-close DQN: 16-32 conv, FC256."""
        def __init__(self, n_actions):
            super().__init__()
            self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
            self.fc1 = nn.Linear(32 * 9 * 9, 256)
            self.fc2 = nn.Linear(256, n_actions)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.relu(self.conv2(x))
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            return self.fc2(x)


class _ReplayBuffer:
    """Shared replay for both DQN variants. Stored as float32 to keep code simple."""
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.idx = 0
        self.full = False
        self.states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.next_states = np.zeros((capacity, *state_shape), dtype=np.float32)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def store(self, s, a, r, s2, d):
        i = self.idx % self.capacity
        self.states[i] = s
        self.next_states[i] = s2
        self.actions[i] = a
        self.rewards[i] = r
        self.dones[i] = d
        self.idx += 1
        if self.idx >= self.capacity:
            self.full = True

    def __len__(self):
        return self.capacity if self.full else self.idx

    def sample(self, batch_size, device):
        size = len(self)
        b = np.random.choice(size, batch_size, replace=False)
        return (
            T.as_tensor(self.states[b], device=device, dtype=T.float32),
            T.as_tensor(self.actions[b], device=device, dtype=T.long),
            T.as_tensor(self.rewards[b], device=device, dtype=T.float32),
            T.as_tensor(self.next_states[b], device=device, dtype=T.float32),
            T.as_tensor(self.dones[b], device=device, dtype=T.bool),
        )


class DQNTask8(Algorithm):
    """DQN as written in Task_8-main/dqn_cnn.py + car_racing.py."""
    name = "dqn"
    needs_loss = True
    needs_epsilon = True

    def __init__(self, lr=1e-4, gamma=0.99, batch_size=64,
                 eps_start=1.0, eps_end=0.01, eps_decay=0.99995,
                 target_update_freq=1_000, learn_every=4):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for DQN.")
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps = eps_start
        self.eps_end = eps_end
        self.eps_decay = eps_decay
        self.target_update_freq = target_update_freq
        self.learn_every = learn_every
        self.frame_size = 48
        self._counts = collections.Counter()

    def _build(self):
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.q_eval = _DQN48(N_DISCRETE).to(self.device)
        self.q_target = _DQN48(N_DISCRETE).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()
        self.optimizer = optim.Adam(self.q_eval.parameters(), lr=self.lr)
        self.loss_fn = nn.MSELoss()

    def _choose(self, state):
        if random.random() < self.eps:
            return random.randrange(N_DISCRETE)
        s = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            q = self.q_eval(s)
        return int(T.argmax(q, dim=1).item())

    def _learn(self, buffer):
        if len(buffer) < self.batch_size:
            return None
        s, a, r, s2, d = buffer.sample(self.batch_size, self.device)
        q_pred = self.q_eval(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with T.no_grad():
            q_next = self.q_target(s2).max(dim=1).values
            q_next[d] = 0.0
            target = r + self.gamma * q_next
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.eps = max(self.eps_end, self.eps * self.eps_decay)
        return float(loss.item())

    def train(self, env, cfg):
        self._build()
        buffer = _ReplayBuffer(cfg.replay_size, (4, self.frame_size, self.frame_size))
        stacker = FrameStack(n=4, size=self.frame_size)
        learn_step = 0
        total_steps = 0
        records: list[EpisodeRecord] = []

        for ep in range(cfg.episodes):
            raw, _ = env.reset(seed=cfg.seed + ep)
            state = stacker.reset(raw)
            done = False
            score = 0.0
            steps = 0
            last_loss = None

            while not done and steps < cfg.max_episode_steps:
                a = self._choose(state)
                self._counts[a] += 1
                raw_next, r, term, trunc, _ = env.step(DISCRETE_ACTIONS[a])
                next_state = stacker.step(raw_next)
                done = term or trunc
                buffer.store(state, a, r, next_state, done)

                total_steps += 1
                steps += 1
                score += r

                if total_steps > cfg.warmup_steps and total_steps % self.learn_every == 0:
                    last_loss = self._learn(buffer)
                    if last_loss is not None:
                        learn_step += 1
                        if learn_step % self.target_update_freq == 0:
                            self.q_target.load_state_dict(self.q_eval.state_dict())

                state = next_state

            records.append(EpisodeRecord(ep, score, steps,
                                         loss=last_loss, epsilon=self.eps))
            _log_episode(self.name, ep, cfg.episodes, score,
                         np.mean([r.score for r in records[-100:]]),
                         extra=f"eps={self.eps:.3f} loss={_fmt(last_loss)}")
        return records

    def action_counts(self):
        return dict(self._counts)


# -----------------------------------------------------------------------------
# DQN-paper (car_racing_DQN.py) — 84x84 CNN, RMSProp, Huber, Double-DQN
# -----------------------------------------------------------------------------
class DQNPaper(Algorithm):
    name = "dqn_paper"
    needs_loss = True
    needs_epsilon = True

    def __init__(self, lr=2.5e-4, gamma=0.99, batch_size=64,
                 eps_warmup_frames=5_000, eps_anneal_frames=200_000,
                 eps_start=1.0, eps_end=0.05, target_update=2_000):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for DQN-paper.")
        self.lr = lr
        self.gamma = gamma
        self.batch_size = batch_size
        self.eps_warmup = eps_warmup_frames
        self.eps_anneal = eps_anneal_frames
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.target_update = target_update
        self.frame_size = 84
        self._counts = collections.Counter()
        self.learn_steps = 0

    def _build(self):
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.q_eval = _DQN84(N_DISCRETE).to(self.device)
        self.q_target = _DQN84(N_DISCRETE).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()
        self.optimizer = optim.RMSprop(
            self.q_eval.parameters(), lr=self.lr,
            alpha=0.95, eps=0.01, momentum=0.0, centered=False,
        )
        self.loss_fn = nn.SmoothL1Loss()

    def _epsilon(self, frame_idx):
        if frame_idx < self.eps_warmup:
            return self.eps_start
        frac = min(1.0, (frame_idx - self.eps_warmup) / float(self.eps_anneal))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def _choose(self, state, frame_idx):
        eps = self._epsilon(frame_idx)
        if random.random() < eps:
            return random.randrange(N_DISCRETE), eps
        s = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0)
        with T.no_grad():
            q = self.q_eval(s)
        return int(T.argmax(q, dim=1).item()), eps

    def _learn(self, buffer):
        if len(buffer) < self.batch_size:
            return None
        s, a, r, s2, d = buffer.sample(self.batch_size, self.device)
        q_pred = self.q_eval(s).gather(1, a.unsqueeze(1)).squeeze(1)
        with T.no_grad():
            # Double-DQN: action selected by q_eval, evaluated by q_target.
            next_a = self.q_eval(s2).argmax(dim=1)
            q_next = self.q_target(s2).gather(1, next_a.unsqueeze(1)).squeeze(1)
            q_next[d] = 0.0
            target = r + self.gamma * q_next
        loss = self.loss_fn(q_pred, target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())
        return float(loss.item())

    def train(self, env, cfg):
        self._build()
        buffer = _ReplayBuffer(cfg.replay_size, (4, self.frame_size, self.frame_size))
        stacker = FrameStack(n=4, size=self.frame_size)
        total_steps = 0
        records: list[EpisodeRecord] = []

        for ep in range(cfg.episodes):
            raw, _ = env.reset(seed=cfg.seed + ep)
            state = stacker.reset(raw)
            done = False
            score = 0.0
            steps = 0
            last_loss = None
            last_eps = self._epsilon(total_steps)

            while not done and steps < cfg.max_episode_steps:
                a, eps = self._choose(state, total_steps)
                self._counts[a] += 1
                raw_next, r, term, trunc, _ = env.step(DISCRETE_ACTIONS[a])
                next_state = stacker.step(raw_next)
                done = term or trunc
                # Reward shaping from the original paper-close run.
                shaped = float(np.tanh(r / 5.0))
                buffer.store(state, a, shaped, next_state, done)

                total_steps += 1
                steps += 1
                score += r
                last_eps = eps

                if total_steps > cfg.warmup_steps:
                    last_loss = self._learn(buffer)

                state = next_state

            records.append(EpisodeRecord(ep, score, steps,
                                         loss=last_loss, epsilon=last_eps))
            _log_episode(self.name, ep, cfg.episodes, score,
                         np.mean([r.score for r in records[-100:]]),
                         extra=f"eps={last_eps:.3f} loss={_fmt(last_loss)}")
        return records

    def action_counts(self):
        return dict(self._counts)


# -----------------------------------------------------------------------------
# PPO (Task 6) — Actor-Critic with continuous actions + GAE
# -----------------------------------------------------------------------------
if _HAS_TORCH:

    class _CNNTrunk(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Conv2d(4, 32, kernel_size=8, stride=4), nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
                nn.Flatten(),
                nn.Linear(64 * 8 * 8, 512), nn.ReLU(),
            )
            for m in self.net:
                if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.orthogonal_(m.weight, gain=nn.init.calculate_gain("relu"))
                    nn.init.zeros_(m.bias)

        def forward(self, x):
            return self.net(x)

    class _Actor(nn.Module):
        def __init__(self, feat=512, act=3):
            super().__init__()
            self.mu = nn.Linear(feat, act)
            self.log_std = nn.Parameter(T.full((act,), -1.0))
            nn.init.orthogonal_(self.mu.weight, gain=0.01)
            with T.no_grad():
                self.mu.bias.copy_(T.tensor([0.0, 2.0, -2.0]))

        def forward(self, feats):
            raw = self.mu(feats)
            steer = T.tanh(raw[:, 0:1])
            gas = T.sigmoid(raw[:, 1:2])
            brake = T.sigmoid(raw[:, 2:3])
            mu = T.cat([steer, gas, brake], dim=-1)
            sigma = self.log_std.exp().clamp(0.1, 1.0).expand_as(mu)
            return mu, sigma

        def get_action(self, feats):
            mu, sigma = self(feats)
            dist = T.distributions.Normal(mu, sigma)
            a = dist.sample()
            lp = dist.log_prob(a).sum(-1)
            return a, lp

        def evaluate(self, feats, actions):
            mu, sigma = self(feats)
            dist = T.distributions.Normal(mu, sigma)
            lp = dist.log_prob(actions).sum(-1)
            ent = dist.entropy().sum(-1)
            return lp, ent

    class _Critic(nn.Module):
        def __init__(self, feat=512):
            super().__init__()
            self.v = nn.Linear(feat, 1)
            nn.init.orthogonal_(self.v.weight, gain=1.0)
            nn.init.zeros_(self.v.bias)

        def forward(self, feats):
            return self.v(feats).squeeze(-1)

    class _ActorCritic(nn.Module):
        def __init__(self):
            super().__init__()
            self.cnn = _CNNTrunk()
            self.actor = _Actor()
            self.critic = _Critic()

        def get_action(self, obs):
            f = self.cnn(obs)
            a, lp = self.actor.get_action(f)
            v = self.critic(f)
            return a, lp, v

        def evaluate(self, obs, actions):
            f = self.cnn(obs)
            lp, ent = self.actor.evaluate(f, actions)
            v = self.critic(f)
            return lp, ent, v


def _compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    last = 0.0
    for t in reversed(range(n)):
        next_v = last_value if t == n - 1 else values[t + 1]
        non_term = 1.0 - dones[t]
        delta = rewards[t] + gamma * next_v * non_term - values[t]
        last = delta + gamma * lam * non_term * last
        adv[t] = last
    return adv, adv + values


class PPOTask6(Algorithm):
    """PPO on a single env (the framework benchmarks one process at a time)."""
    name = "ppo"
    needs_loss = True
    needs_epsilon = False

    def __init__(self, lr=2.5e-4, gamma=0.99, gae_lambda=0.95,
                 n_steps=512, ppo_epochs=4, mini_batch=128,
                 clip_eps=0.2, value_coef=0.5, entropy_coef=0.01,
                 max_grad_norm=0.5):
        if not _HAS_TORCH:
            raise RuntimeError("PyTorch is required for PPO.")
        self.lr = lr
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.n_steps = n_steps
        self.ppo_epochs = ppo_epochs
        self.mini_batch = mini_batch
        self.clip_eps = clip_eps
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.frame_size = 96

    def _build(self):
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")
        self.model = _ActorCritic().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)

    def _update(self, batch):
        obs = T.as_tensor(batch["obs"], device=self.device, dtype=T.float32)
        actions = T.as_tensor(batch["actions"], device=self.device, dtype=T.float32)
        returns = T.as_tensor(batch["returns"], device=self.device, dtype=T.float32)
        adv = T.as_tensor(batch["adv"], device=self.device, dtype=T.float32)
        old_lp = T.as_tensor(batch["log_probs"], device=self.device, dtype=T.float32)

        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
        B = obs.shape[0]

        sums = {"actor": 0.0, "critic": 0.0, "entropy": 0.0, "total": 0.0}
        n = 0
        for _ in range(self.ppo_epochs):
            perm = T.randperm(B, device=self.device)
            for s in range(0, B, self.mini_batch):
                idx = perm[s:s + self.mini_batch]
                new_lp, ent, val = self.model.evaluate(obs[idx], actions[idx])
                ratio = (new_lp - old_lp[idx]).exp()
                s1 = ratio * adv[idx]
                s2 = ratio.clamp(1 - self.clip_eps, 1 + self.clip_eps) * adv[idx]
                actor_loss = -T.min(s1, s2).mean()
                critic_loss = F.smooth_l1_loss(val, returns[idx])
                entropy_loss = -ent.mean()
                loss = (actor_loss
                        + self.value_coef * critic_loss
                        + self.entropy_coef * entropy_loss)
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
                self.optimizer.step()

                sums["actor"] += actor_loss.item()
                sums["critic"] += critic_loss.item()
                sums["entropy"] += -entropy_loss.item()
                sums["total"] += loss.item()
                n += 1
        return {k: v / max(n, 1) for k, v in sums.items()}

    def train(self, env, cfg):
        self._build()
        stacker = FrameStack(n=4, size=self.frame_size)
        raw, _ = env.reset(seed=cfg.seed)
        obs = stacker.reset(raw)

        total_steps = 0
        ep_idx = 0
        ep_score = 0.0
        ep_steps = 0
        ep_records: list[EpisodeRecord] = []
        # PPO can't easily attribute a single "loss" to a single episode; we
        # carry the most-recent update's losses forward.
        last_losses = {"total": None, "actor": None, "critic": None, "entropy": None}

        budget = min(cfg.ppo_total_steps, cfg.episodes * cfg.max_episode_steps * 2)

        while total_steps < budget and ep_idx < cfg.episodes:
            # ── Rollout ──
            obs_buf = np.zeros((self.n_steps, 4, self.frame_size, self.frame_size),
                               dtype=np.float32)
            act_buf = np.zeros((self.n_steps, 3), dtype=np.float32)
            rew_buf = np.zeros(self.n_steps, dtype=np.float32)
            done_buf = np.zeros(self.n_steps, dtype=np.float32)
            val_buf = np.zeros(self.n_steps, dtype=np.float32)
            lp_buf = np.zeros(self.n_steps, dtype=np.float32)

            for t in range(self.n_steps):
                obs_t = T.as_tensor(obs, device=self.device, dtype=T.float32).unsqueeze(0)
                with T.no_grad():
                    a, lp, v = self.model.get_action(obs_t)
                a_np = a.cpu().numpy()[0]
                a_np[0] = float(np.clip(a_np[0], -1.0, 1.0))
                a_np[1] = float(np.clip(a_np[1], 0.0, 1.0))
                a_np[2] = float(np.clip(a_np[2], 0.0, 1.0))

                raw_next, r, term, trunc, _ = env.step(a_np)
                done = term or trunc

                obs_buf[t] = obs
                act_buf[t] = a_np
                rew_buf[t] = r
                done_buf[t] = float(done)
                val_buf[t] = v.item()
                lp_buf[t] = lp.item()

                ep_score += r
                ep_steps += 1
                total_steps += 1

                if done or ep_steps >= cfg.max_episode_steps:
                    ep_records.append(EpisodeRecord(
                        ep_idx, ep_score, ep_steps,
                        loss=last_losses["total"],
                        extra={k: v for k, v in last_losses.items() if v is not None},
                    ))
                    _log_episode(self.name, ep_idx, cfg.episodes, ep_score,
                                 np.mean([rr.score for rr in ep_records[-100:]]),
                                 extra=f"loss={_fmt(last_losses['total'])}")
                    ep_idx += 1
                    ep_score = 0.0
                    ep_steps = 0
                    raw, _ = env.reset(seed=cfg.seed + ep_idx)
                    obs = stacker.reset(raw)
                    if ep_idx >= cfg.episodes:
                        break
                else:
                    obs = stacker.step(raw_next)

            # Truncate buffers to actually-filled length.
            used = t + 1
            obs_buf = obs_buf[:used]
            act_buf = act_buf[:used]
            rew_buf = rew_buf[:used]
            done_buf = done_buf[:used]
            val_buf = val_buf[:used]
            lp_buf = lp_buf[:used]

            # Bootstrap value for the trailing state.
            obs_t = T.as_tensor(obs, device=self.device, dtype=T.float32).unsqueeze(0)
            with T.no_grad():
                _, _, last_v = self.model.get_action(obs_t)
            last_v = last_v.item()

            adv, ret = _compute_gae(rew_buf, val_buf, done_buf, last_v,
                                    self.gamma, self.gae_lambda)
            losses = self._update({
                "obs": obs_buf, "actions": act_buf, "returns": ret,
                "adv": adv, "log_probs": lp_buf,
            })
            last_losses["total"] = losses["total"]
            last_losses["actor"] = losses["actor"]
            last_losses["critic"] = losses["critic"]
            last_losses["entropy"] = losses["entropy"]

        return ep_records


# =============================================================================
# Algorithm registry + runner
# =============================================================================

REGISTRY: dict[str, Callable[[], Algorithm]] = {
    "random": RandomAgent,
}
if _HAS_TORCH:
    REGISTRY["dqn"] = DQNTask8
    REGISTRY["dqn_paper"] = DQNPaper
    REGISTRY["ppo"] = PPOTask6


def _fmt(x):
    return "—" if x is None else f"{x:.4f}"


def _log_episode(algo: str, ep: int, total: int, score: float,
                 avg100: float, extra: str = "") -> None:
    print(f"  [{algo:<10}] ep {ep+1:>4}/{total}  "
          f"score={score:>8.2f}  avg100={avg100:>8.2f}  {extra}",
          flush=True)


def _make_env(cfg: BenchmarkConfig):
    if not _HAS_GYM:
        raise RuntimeError("gymnasium is required to run the benchmark.")
    return gym.make(cfg.env_name)


def run_algorithm(name: str, cfg: BenchmarkConfig
                  ) -> tuple[list[EpisodeRecord], float, Algorithm]:
    print(f"\n=== Training {name} ({cfg.mode} mode, {cfg.episodes} episodes) ===")
    if name not in REGISTRY:
        raise KeyError(f"Unknown algorithm: {name}. "
                       f"Choose from {sorted(REGISTRY)}")
    algo = REGISTRY[name]()
    env = _make_env(cfg)
    try:
        t0 = time.time()
        records = algo.train(env, cfg)
    finally:
        env.close()
    elapsed = time.time() - t0
    print(f"=== {name} done in {elapsed/60:.1f} min "
          f"({len(records)} episodes recorded) ===")
    return records, elapsed, algo


# =============================================================================
# Plotting
# =============================================================================

def _rolling(xs, n=100):
    out = np.empty(len(xs))
    for i in range(len(xs)):
        out[i] = np.mean(xs[max(0, i - n + 1):i + 1])
    return out


def plot_rewards(results: dict[str, list[EpisodeRecord]], path: str) -> None:
    if not _HAS_MPL:
        print("[plots] matplotlib missing — skipping reward plot.")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    for name, recs in results.items():
        if not recs:
            continue
        xs = np.array([r.episode for r in recs])
        ys = np.array([r.score for r in recs])
        ax1.plot(xs, ys, label=name, alpha=0.5)
        ax2.plot(xs, _rolling(ys, 100), label=name, linewidth=2)
    ax1.set_title("Per-episode reward")
    ax1.set_xlabel("Episode"); ax1.set_ylabel("Score"); ax1.legend(); ax1.grid(alpha=0.3)
    ax2.set_title("Rolling-100 reward")
    ax2.set_xlabel("Episode"); ax2.set_ylabel("Score"); ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def plot_losses(results: dict[str, list[EpisodeRecord]], path: str) -> None:
    if not _HAS_MPL:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = False
    for name, recs in results.items():
        losses = [(r.episode, r.loss) for r in recs if r.loss is not None]
        if not losses:
            continue
        xs, ys = zip(*losses)
        ax.plot(xs, ys, label=name)
        plotted = True
    if not plotted:
        plt.close(fig); return
    ax.set_title("Training loss per episode (last-step within episode)")
    ax.set_xlabel("Episode"); ax.set_ylabel("Loss"); ax.set_yscale("log")
    ax.legend(); ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def plot_policy(algos: dict[str, Algorithm], path: str) -> None:
    if not _HAS_MPL:
        return
    items = [(name, a.action_counts()) for name, a in algos.items()
             if a.action_counts()]
    if not items:
        return
    action_labels = ["noop", "left", "right", "gas", "brake", "gas+L", "gas+R"]
    fig, ax = plt.subplots(figsize=(11, 5))
    width = 0.8 / len(items)
    x = np.arange(N_DISCRETE)
    for i, (name, counts) in enumerate(items):
        total = sum(counts.values()) or 1
        freqs = [counts.get(j, 0) / total for j in range(N_DISCRETE)]
        ax.bar(x + i * width, freqs, width, label=name)
    ax.set_title("Action distribution across training (discrete-action agents)")
    ax.set_xticks(x + width * (len(items) - 1) / 2)
    ax.set_xticklabels(action_labels)
    ax.set_ylabel("Frequency"); ax.legend(); ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] wrote {path}")


def plot_ppo_components(results: dict[str, list[EpisodeRecord]], path: str) -> None:
    if not _HAS_MPL or "ppo" not in results:
        return
    recs = [r for r in results["ppo"] if r.extra]
    if not recs:
        return
    xs = [r.episode for r in recs]
    actor = [r.extra.get("actor") for r in recs]
    critic = [r.extra.get("critic") for r in recs]
    entropy = [r.extra.get("entropy") for r in recs]
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(xs, actor); axes[0].set_title("PPO actor loss"); axes[0].grid(alpha=0.3)
    axes[1].plot(xs, critic); axes[1].set_title("PPO critic loss"); axes[1].grid(alpha=0.3)
    axes[2].plot(xs, entropy); axes[2].set_title("PPO entropy"); axes[2].grid(alpha=0.3)
    for a in axes:
        a.set_xlabel("Episode")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    print(f"[plots] wrote {path}")


# =============================================================================
# Reporting
# =============================================================================

def write_leaderboard(results: dict[str, list[EpisodeRecord]],
                      timings: dict[str, float],
                      path: str) -> None:
    rows = []
    for name, recs in results.items():
        if not recs:
            continue
        scores = np.array([r.score for r in recs])
        last100 = scores[-100:] if len(scores) >= 100 else scores
        rows.append({
            "algorithm": name,
            "episodes": int(len(scores)),
            "final_mean_last100": float(np.mean(last100)),
            "final_std_last100": float(np.std(last100)),
            "best_episode_score": float(np.max(scores)),
            "overall_mean": float(np.mean(scores)),
            "wallclock_minutes": round(timings.get(name, 0.0) / 60.0, 2),
        })
    rows.sort(key=lambda r: r["final_mean_last100"], reverse=True)

    with open(path, "w") as f:
        f.write("rank,algorithm,episodes,final_mean_last100,final_std_last100,"
                "best_episode_score,overall_mean,wallclock_minutes\n")
        for i, r in enumerate(rows, 1):
            f.write(f"{i},{r['algorithm']},{r['episodes']},"
                    f"{r['final_mean_last100']:.3f},{r['final_std_last100']:.3f},"
                    f"{r['best_episode_score']:.3f},{r['overall_mean']:.3f},"
                    f"{r['wallclock_minutes']}\n")

    print("\n=== Leaderboard ===")
    print(f"{'rank':<5}{'algo':<12}{'avg_last100':>14}{'best':>10}{'time (m)':>12}")
    for i, r in enumerate(rows, 1):
        print(f"{i:<5}{r['algorithm']:<12}{r['final_mean_last100']:>14.2f}"
              f"{r['best_episode_score']:>10.2f}{r['wallclock_minutes']:>12.2f}")


def write_run_log(results: dict[str, list[EpisodeRecord]], path: str) -> None:
    serialised = {
        name: [asdict(r) for r in recs] for name, recs in results.items()
    }
    with open(path, "w") as f:
        json.dump(serialised, f, indent=2)


# =============================================================================
# Main
# =============================================================================

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="One-click RL benchmark for CarRacing-v3.")
    p.add_argument("--mode", choices=["smoke", "medium", "full"], default="smoke",
                   help="Run budget. smoke ≈ 3-5 min/algo, medium ≈ 30 min/algo, "
                        "full ≈ overnight.")
    p.add_argument("--algos", nargs="+", default=None,
                   choices=["random", "dqn", "dqn_paper", "ppo"],
                   help="Subset of algorithms to benchmark. Default: all that "
                        "are available given installed dependencies "
                        "(torch is required for dqn / dqn_paper / ppo).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-dir", default="results")
    p.add_argument("--max-episode-steps", type=int, default=1_000,
                   help="Cap per-episode steps to keep wallclock predictable.")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Determinism (best-effort).
    random.seed(args.seed); np.random.seed(args.seed)
    if _HAS_TORCH:
        T.manual_seed(args.seed)

    # Friendly preflight.
    missing = []
    if not _HAS_TORCH: missing.append("torch")
    if not _HAS_GYM: missing.append("gymnasium[box2d]")
    if not _HAS_MPL: missing.append("matplotlib")
    if not _HAS_CV2: missing.append("opencv-python (optional, faster preprocessing)")
    if missing:
        print(f"[warn] Missing packages: {', '.join(missing)}")
        print("       Install: pip install torch gymnasium[box2d] matplotlib opencv-python")
    if not _HAS_GYM:
        print("[fatal] gymnasium is required to run the benchmark.")
        return 2

    cfg = BenchmarkConfig(
        mode=args.mode, seed=args.seed, out_dir=args.out_dir,
        max_episode_steps=args.max_episode_steps,
    )

    algos = args.algos or [k for k in ("random", "dqn", "dqn_paper", "ppo")
                           if k in REGISTRY]
    # If torch is unavailable, drop torch-dependent algorithms with a warning
    # rather than crashing — the random baseline still runs and validates the
    # rest of the pipeline.
    if not _HAS_TORCH:
        torch_algos = {"dqn", "dqn_paper", "ppo"}
        dropped = [a for a in algos if a in torch_algos]
        if dropped:
            print(f"[warn] Skipping {dropped} — PyTorch is not installed.")
        algos = [a for a in algos if a not in torch_algos]
        if not algos:
            print("[fatal] No algorithms left to run after dropping torch-only ones.")
            return 2
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = os.path.join(args.out_dir, f"benchmark_{cfg.mode}_{timestamp}")
    os.makedirs(out, exist_ok=True)
    print(f"[setup] mode={cfg.mode}  algos={algos}  out={out}")
    print(f"[setup] device={'cuda' if (_HAS_TORCH and T.cuda.is_available()) else 'cpu'}")

    # Persist run config for reproducibility before we start training.
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump({
            "mode": cfg.mode, "seed": cfg.seed, "env": cfg.env_name,
            "episodes": cfg.episodes, "replay_size": cfg.replay_size,
            "ppo_total_steps": cfg.ppo_total_steps, "warmup_steps": cfg.warmup_steps,
            "max_episode_steps": cfg.max_episode_steps,
            "algorithms": algos,
            "timestamp": timestamp,
            "torch_version": T.__version__ if _HAS_TORCH else None,
            "cuda": bool(_HAS_TORCH and T.cuda.is_available()),
        }, f, indent=2)

    results: dict[str, list[EpisodeRecord]] = {}
    instances: dict[str, Algorithm] = {}
    timings: dict[str, float] = {}

    for name in algos:
        try:
            recs, elapsed, algo = run_algorithm(name, cfg)
            results[name] = recs
            instances[name] = algo
            timings[name] = elapsed
            # Write the run log as we go so a crash mid-benchmark doesn't lose data.
            write_run_log(results, os.path.join(out, "run_log.json"))
        except Exception as e:  # noqa: BLE001
            print(f"[error] {name} crashed: {e!r} — continuing with remaining algos.")
            results[name] = []

    plot_rewards(results, os.path.join(out, "reward_curves.png"))
    plot_losses(results, os.path.join(out, "loss_curves.png"))
    plot_policy(instances, os.path.join(out, "policy_distribution.png"))
    plot_ppo_components(results, os.path.join(out, "ppo_components.png"))
    write_leaderboard(results, timings, os.path.join(out, "leaderboard.csv"))

    print(f"\n[done] All artifacts saved to: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
