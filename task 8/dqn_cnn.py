import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np


class DeepQCNN(nn.Module):
    """DQN CNN architecture for 48x48 image input."""
    def __init__(self, lr, n_actions, input_channels=4):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        # 48 -> 11 -> 4 -> 2
        self.fc1 = nn.Linear(64 * 2 * 2, 512)
        self.fc2 = nn.Linear(512, n_actions)

        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.loss = nn.MSELoss()
        self.device = T.device('cuda:0' if T.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


class AgentCNN:
    def __init__(self, gamma, epsilon, lr, n_actions, input_shape,
                 batch_size, max_mem_size=10_000, eps_end=0.01,
                 eps_decay=0.99995, target_update_freq=1000):
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_min = eps_end
        self.eps_decay = eps_decay
        self.n_actions = n_actions
        self.action_space = list(range(n_actions))
        self.mem_size = max_mem_size
        self.batch_size = batch_size
        self.mem_cntr = 0
        self.learn_step_cntr = 0
        self.target_update_freq = target_update_freq

        self.Q_eval = DeepQCNN(lr, n_actions=n_actions,
                               input_channels=input_shape[0])

        self.Q_target = DeepQCNN(lr, n_actions=n_actions,
                                 input_channels=input_shape[0])
        self.Q_target.load_state_dict(self.Q_eval.state_dict())
        self.Q_target.eval()

        self.state_memory = np.zeros((max_mem_size, *input_shape),
                                     dtype=np.float32)
        self.new_state_memory = np.zeros((max_mem_size, *input_shape),
                                           dtype=np.float32)
        self.action_memory = np.zeros(max_mem_size, dtype=np.int32)
        self.reward_memory = np.zeros(max_mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(max_mem_size, dtype=np.bool_)

    def store_transition(self, state, action, reward, state_, done):
        idx = self.mem_cntr % self.mem_size
        self.state_memory[idx] = state
        self.new_state_memory[idx] = state_
        self.reward_memory[idx] = reward
        self.action_memory[idx] = action
        self.terminal_memory[idx] = done
        self.mem_cntr += 1

    def choose_action(self, observation):
        if np.random.random() > self.epsilon:
            state = T.as_tensor(
                observation, dtype=T.float32, device=self.Q_eval.device
            ).unsqueeze(0)
            with T.no_grad():
                q_values = self.Q_eval(state)
            return T.argmax(q_values).item()
        return np.random.choice(self.action_space)

    def learn(self):
        if self.mem_cntr < self.batch_size:
            return

        self.Q_eval.optimizer.zero_grad()

        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, self.batch_size, replace=False)
        batch_idx = np.arange(self.batch_size, dtype=np.int32)

        states = T.tensor(self.state_memory[batch]).to(self.Q_eval.device)
        new_states = T.tensor(self.new_state_memory[batch]).to(self.Q_eval.device)
        rewards = T.tensor(self.reward_memory[batch]).to(self.Q_eval.device)
        terminals = T.tensor(self.terminal_memory[batch]).to(self.Q_eval.device)
        actions = self.action_memory[batch]

        q_eval = self.Q_eval(states)[batch_idx, actions]

        with T.no_grad():
            q_next = self.Q_target(new_states)
        q_next[terminals] = 0.0

        q_target = rewards + self.gamma * T.max(q_next, dim=1)[0]

        loss = self.Q_eval.loss(q_target, q_eval)
        loss.backward()
        self.Q_eval.optimizer.step()

        self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)

        self.learn_step_cntr += 1
        if self.learn_step_cntr % self.target_update_freq == 0:
            self.Q_target.load_state_dict(self.Q_eval.state_dict())
