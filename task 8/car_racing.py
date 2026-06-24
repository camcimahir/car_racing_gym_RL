import collections
import time
import gymnasium as gym
import numpy as np
import cv2

from dqn_cnn import AgentCNN
from utils1 import plot_learning_curve

# CarRacing-v3 has continuous actions: [steering, gas, brake].
# DQN needs discrete actions, so we map 5 integers to fixed control vectors.
DISCRETE_ACTIONS = [
    np.array([0.0, 0.0, 0.0]),   # 0: do nothing
    np.array([-1.0, 0.0, 0.0]),  # 1: steer left
    np.array([1.0, 0.0, 0.0]),   # 2: steer right
    np.array([0.0, 1.0, 0.0]),   # 3: gas
    np.array([0.0, 0.0, 0.8]),   # 4: brake
]


class SkipFrame(gym.Wrapper):
    """Repeat the chosen action for `skip` frames, accumulating reward."""
    def __init__(self, env, skip=4):
        super().__init__(env)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        for _ in range(self._skip):
            state, reward, terminated, truncated, info = self.env.step(action)
            total_reward += reward
            if terminated:
                break
        return state, total_reward, terminated, truncated, info


def preprocess_frame(frame):
    """RGB (96,96,3) -> grayscale (48,48), normalised to [0,1]."""
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_AREA)
    return resized.astype(np.float32) / 255.0


class FrameStack:
    """Stacks the last `n` preprocessed frames into a single observation."""
    def __init__(self, n=4):
        self.n = n
        self.frames = collections.deque(maxlen=n)

    def reset(self, raw_frame):
        processed = preprocess_frame(raw_frame)
        for _ in range(self.n):
            self.frames.append(processed)
        return np.array(self.frames)          # (4, 48, 48)

    def step(self, raw_frame):
        self.frames.append(preprocess_frame(raw_frame))
        return np.array(self.frames)           # (4, 48, 48)


if __name__ == '__main__':
    env = gym.make('CarRacing-v3')
    env = SkipFrame(env, skip=4)
    frame_stack = FrameStack(n=4)

    agent = AgentCNN(
        gamma=0.99,
        epsilon=1.0,
        lr=1e-4,
        n_actions=len(DISCRETE_ACTIONS),
        input_shape=(4, 48, 48),
        batch_size=64,
        max_mem_size=10_000,
        eps_end=0.01,
        eps_decay=0.99995,
        target_update_freq=1000,
    )

    scores, eps_history = [], []
    n_games = 1000
    total_steps = 0
    WARMUP = 1000
    LEARN_EVERY = 4

    for i in range(n_games):
        score = 0
        done = False
        steps = 0
        ep_start = time.time()
        raw_obs, _ = env.reset()
        observation = frame_stack.reset(raw_obs)

        while not done:
            action_idx = agent.choose_action(observation)
            raw_obs_, reward, terminated, truncated, info = env.step(
                DISCRETE_ACTIONS[action_idx]
            )
            done = terminated or truncated
            steps += 1
            total_steps += 1

            observation_ = frame_stack.step(raw_obs_)
            score += reward

            agent.store_transition(observation, action_idx, reward,
                                   observation_, done)

            if total_steps > WARMUP and total_steps % LEARN_EVERY == 0:
                agent.learn()

            observation = observation_

        ep_time = time.time() - ep_start
        fps = steps / ep_time if ep_time > 0 else 0

        scores.append(score)
        eps_history.append(agent.epsilon)

        avg_score = np.mean(scores[-100:])
        print(f'episode: {i}  score: {score:.2f}  '
              f'avg score: {avg_score:.2f}  epsilon: {agent.epsilon:.2f}  '
              f'steps: {steps}  fps: {fps:.0f}')

    env.close()
    x = list(range(1, n_games + 1))
    plot_learning_curve(x, scores, eps_history, 'car_racing_v2.png')
