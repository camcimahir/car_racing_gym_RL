import collections
import os
import random
import time
from datetime import datetime

import cv2
import gymnasium as gym
import numpy as np
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0], dtype=np.float32),
    np.array([-1.0, 0.0, 0.0], dtype=np.float32),
    np.array([1.0, 0.0, 0.0], dtype=np.float32),
    np.array([0.0, 1.0, 0.0], dtype=np.float32),
    np.array([-1.0, 1.0, 0.0], dtype=np.float32),
    np.array([1.0, 1.0, 0.0], dtype=np.float32),
    np.array([0.0, 0.0, 0.8], dtype=np.float32),
]


class SkipFrame(gym.Wrapper):
    """Repeat an action for `skip` steps and accumulate reward."""

    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        terminated = False
        truncated = False
        info = {}
        obs = None
        for _ in range(self.skip):
            obs, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        return obs, total_reward, terminated, truncated, info


def preprocess_frame(frame):
    """
    DQN-style preprocessing:
    - RGB -> grayscale
    - Resize to (110, 84)
    - Crop center play area to (84, 84)
    Returns uint8 for replay-memory efficiency.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (84, 110), interpolation=cv2.INTER_AREA)
    cropped = resized[18:102, :]
    return cropped.astype(np.uint8)


class FrameStack:
    """Stack last n preprocessed frames into (n, 84, 84)."""

    def __init__(self, n=4):
        self.n = n
        self.frames = collections.deque(maxlen=n)

    def reset(self, raw_frame):
        frame = preprocess_frame(raw_frame)
        self.frames.clear()
        for _ in range(self.n):
            self.frames.append(frame)
        return np.stack(self.frames, axis=0)

    def step(self, raw_frame):
        self.frames.append(preprocess_frame(raw_frame))
        return np.stack(self.frames, axis=0)


class DQNNet(nn.Module):
    """
    Paper-close architecture from the attached DQN preprint:
    4x84x84 -> Conv(16, 8x8, s4) -> Conv(32, 4x4, s2) -> FC256 -> actions.
    """

    def __init__(self, n_actions):
        super().__init__()
        self.conv1 = nn.Conv2d(4, 16, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=4, stride=2)
        self.fc1 = nn.Linear(32 * 9 * 9, 256)
        self.fc2 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = T.relu(self.conv1(x))
        x = T.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = T.relu(self.fc1(x))
        return self.fc2(x)


class ReplayBuffer:
    def __init__(self, capacity, state_shape):
        self.capacity = capacity
        self.mem_cntr = 0

        self.state_memory = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.next_state_memory = np.zeros((capacity, *state_shape), dtype=np.uint8)
        self.action_memory = np.zeros(capacity, dtype=np.int64)
        self.reward_memory = np.zeros(capacity, dtype=np.float32)
        self.done_memory = np.zeros(capacity, dtype=np.bool_)

    def store(self, state, action, reward, next_state, done):
        idx = self.mem_cntr % self.capacity
        self.state_memory[idx] = state
        self.next_state_memory[idx] = next_state
        self.action_memory[idx] = action
        self.reward_memory[idx] = reward
        self.done_memory[idx] = done
        self.mem_cntr += 1

    def sample(self, batch_size, device):
        max_mem = min(self.mem_cntr, self.capacity)
        batch = np.random.choice(max_mem, batch_size, replace=False)

        states = T.as_tensor(self.state_memory[batch], device=device, dtype=T.float32) / 255.0
        next_states = T.as_tensor(self.next_state_memory[batch], device=device, dtype=T.float32) / 255.0
        actions = T.as_tensor(self.action_memory[batch], device=device, dtype=T.long)
        rewards = T.as_tensor(self.reward_memory[batch], device=device, dtype=T.float32)
        dones = T.as_tensor(self.done_memory[batch], device=device, dtype=T.bool)
        return states, actions, rewards, next_states, dones


class Agent:
    def __init__(
        self,
        n_actions,
        gamma=0.99,
        lr=0.00025,
        batch_size=32,
        replay_size=1_000_000,
        target_update=10_000,
        eps_start=1.0,
        eps_end=0.1,
        eps_warmup_frames=20_000,
        eps_anneal_frames=1_000_000,
    ):
        self.n_actions = n_actions
        self.gamma = gamma
        self.batch_size = batch_size
        self.target_update = target_update

        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_warmup_frames = eps_warmup_frames
        self.eps_anneal_frames = eps_anneal_frames

        self.learn_steps = 0
        self.device = T.device("cuda:0" if T.cuda.is_available() else "cpu")

        self.q_eval = DQNNet(n_actions).to(self.device)
        self.q_target = DQNNet(n_actions).to(self.device)
        self.q_target.load_state_dict(self.q_eval.state_dict())
        self.q_target.eval()

        # Paper-close optimizer/loss choices.
        self.optimizer = optim.RMSprop(
            self.q_eval.parameters(),
            lr=lr,
            alpha=0.95,
            eps=0.01,
            momentum=0.0,
            centered=False,
        )
        self.loss_fn = nn.SmoothL1Loss()

        self.replay = ReplayBuffer(replay_size, state_shape=(4, 84, 84))

    def epsilon(self, frame_idx):
        if frame_idx < self.eps_warmup_frames:
            return self.eps_start
        frac = min(1.0, (frame_idx - self.eps_warmup_frames) / float(self.eps_anneal_frames))
        return self.eps_start + frac * (self.eps_end - self.eps_start)

    def choose_action(self, state, frame_idx):
        eps = self.epsilon(frame_idx)
        if random.random() < eps:
            return random.randrange(self.n_actions), eps

        s = T.as_tensor(state, dtype=T.float32, device=self.device).unsqueeze(0) / 255.0
        with T.no_grad():
            q_values = self.q_eval(s)
        return int(T.argmax(q_values, dim=1).item()), eps

    def learn(self):
        if self.replay.mem_cntr < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay.sample(
            self.batch_size, self.device
        )

        q_pred = self.q_eval(states).gather(1, actions.unsqueeze(1)).squeeze(1)

        with T.no_grad():
            next_actions = self.q_eval(next_states).argmax(dim=1)
            q_next = self.q_target(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            q_next[dones] = 0.0
            q_target = rewards + self.gamma * q_next

        loss = self.loss_fn(q_pred, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.q_eval.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.learn_steps += 1
        if self.learn_steps % self.target_update == 0:
            self.q_target.load_state_dict(self.q_eval.state_dict())

        return float(loss.item())


def main():
    episodes = 1000
    seed = 0

    random.seed(seed)
    np.random.seed(seed)
    T.manual_seed(seed)

    if T.cuda.is_available():
        print(f"CUDA: ON | device: {T.cuda.get_device_name(0)} | torch: {T.__version__}")
    else:
        print(f"CUDA: OFF | using CPU | torch: {T.__version__}")

    # Set up TensorBoard logging (one subfolder per run to avoid merged plots).
    base_log_dir = os.path.expanduser("~/Projects/logs/car_racing_paper_close")
    run_name = datetime.now().strftime("run_%Y%m%d_%H%M%S")
    log_dir = os.path.join(base_log_dir, run_name)
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir)
    print(f"TensorBoard base logs: {base_log_dir}")
    print(f"TensorBoard run logs: {log_dir}")

    env = gym.make("CarRacing-v3")

    frame_stack = FrameStack(n=4)

    agent = Agent(
        n_actions=len(DISCRETE_ACTIONS),
        gamma=0.99,
        lr=0.00025,
        batch_size=64,
        replay_size=300_000,
        target_update=2_000,
        eps_start=1.0,
        eps_end=0.05,
        eps_warmup_frames=20_000,
        eps_anneal_frames=600_000,
    )

    print(f"Agent device: {agent.device}")

    total_steps = 0
    scores = []
    replay_start_size = 5_000
    learn_every = 1

    for ep in range(episodes):
        raw_obs, _ = env.reset(seed=seed + ep)
        state = frame_stack.reset(raw_obs)

        done = False
        ep_score = 0.0
        ep_steps = 0
        ep_start = time.time()
        last_loss = None

        while not done:
            action_idx, eps = agent.choose_action(state, total_steps)

            raw_next, reward, terminated, truncated, _ = env.step(DISCRETE_ACTIONS[action_idx])
            done = terminated or truncated
            next_state = frame_stack.step(raw_next)

            # Keep signal bounded but informative across the large CarRacing reward range.
            shaped_reward = float(np.tanh(reward / 5.0))

            agent.replay.store(state, action_idx, shaped_reward, next_state, done)

            total_steps += 1
            ep_steps += 1
            ep_score += reward

            if (
                total_steps >= replay_start_size
                and total_steps % learn_every == 0
            ):
                last_loss = agent.learn()

            state = next_state

        elapsed = time.time() - ep_start
        fps = ep_steps / elapsed if elapsed > 0 else 0.0
        scores.append(ep_score)
        avg_100 = float(np.mean(scores[-100:]))

        msg = (
            f"episode: {ep}  score: {ep_score:.2f}  avg100: {avg_100:.2f}  "
            f"epsilon: {eps:.3f}  steps: {ep_steps}  fps: {fps:.0f}"
        )
        if last_loss is not None:
            msg += f"  loss: {last_loss:.4f}"
        print(msg)

        # Log metrics to TensorBoard
        writer.add_scalar("episode_score", ep_score, ep)
        writer.add_scalar("avg100_score", avg_100, ep)
        writer.add_scalar("epsilon", eps, ep)
        writer.add_scalar("fps", fps, ep)
        writer.add_scalar("episode_steps", ep_steps, ep)
        if last_loss is not None:
            writer.add_scalar("loss", last_loss, ep)

    writer.flush()
    writer.close()
    env.close()


if __name__ == "__main__":
    main()
