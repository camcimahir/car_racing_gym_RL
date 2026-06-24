import numpy as np
import torch
from collections import deque


FRAME_SIZE = 96


def preprocess_frame(obs):
    """
    Convert raw 96x96x3 frame to grayscale 96x96 normalized float.
    """
    gray = 0.2989 * obs[:, :, 0] + 0.5870 * obs[:, :, 1] + 0.1140 * obs[:, :, 2]
    gray = gray.astype(np.float32) / 255.0
    return gray


class FrameStack:

    def __init__(self, n=4):
        self.n = n
        self.frames = deque(maxlen=n)

    def reset(self, obs):
        frame = preprocess_frame(obs)
        # Fill all slots with the first frame
        for _ in range(self.n):
            self.frames.append(frame)
        return self._get_obs()

    def step(self, obs):
        self.frames.append(preprocess_frame(obs))
        return self._get_obs()

    def _get_obs(self):
        return np.array(self.frames, dtype=np.float32)  # (4, 96, 96)


def to_tensor(x, device):
    return torch.FloatTensor(x).to(device)


def compute_returns(rewards, dones, last_value, gamma=0.99):

    returns = []
    R = last_value
    for reward, done in zip(reversed(rewards), reversed(dones)):
        R = reward + gamma * R * (1.0 - done)
        returns.insert(0, R)
    return returns


def compute_gae(rewards, values, dones, last_value, gamma=0.99, lam=0.95):

    n_steps = len(rewards)
    advantages = np.zeros(n_steps, dtype=np.float32)
    last_gae = 0.0

    for t in reversed(range(n_steps)):
        next_value    = last_value if t == n_steps - 1 else values[t + 1]
        non_terminal  = 1.0 - dones[t]
        delta         = rewards[t] + gamma * next_value * non_terminal - values[t]
        last_gae      = delta + gamma * lam * non_terminal * last_gae
        advantages[t] = last_gae

    returns = advantages + values
    return advantages, returns


# Discrete action set for CarRacing-v3
# can also be found in the GYM documentation
DISCRETE_ACTIONS = [
    [0.0,  0.0, 0.0],   # 0: do nothing
    [-1.0, 0.0, 0.0],   # 1: steer left
    [1.0,  0.0, 0.0],   # 2: steer right
    [0.0,  1.0, 0.0],   # 3: gas
    [0.0,  0.0, 0.8],   # 4: brake
]
DISCRETE_ACTIONS = np.array(DISCRETE_ACTIONS, dtype=np.float32)
NUM_ACTIONS = len(DISCRETE_ACTIONS)
