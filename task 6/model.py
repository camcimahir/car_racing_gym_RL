import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 512),
            nn.ReLU(),
        )
        for module in self.net:
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=nn.init.calculate_gain('relu'))
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.net(x)

class Actor(nn.Module):

    def __init__(self, feature_dim=512, action_dim=3):
        super().__init__()
        self.mu_head = nn.Linear(feature_dim, action_dim)
        self.log_std = nn.Parameter(torch.full((action_dim,), -1.0))

        nn.init.orthogonal_(self.mu_head.weight, gain=0.01)
        with torch.no_grad():
            self.mu_head.bias.copy_(torch.tensor([0.0, 2.0, -2.0]))

    def forward(self, features):
        raw_mu = self.mu_head(features)
        steer = torch.tanh(raw_mu[:, 0:1])
        gas   = torch.sigmoid(raw_mu[:, 1:2])
        brake = torch.sigmoid(raw_mu[:, 2:3])
        mu    = torch.cat([steer, gas, brake], dim=-1)
        sigma = self.log_std.exp().clamp(min=0.1, max=1.0).expand_as(mu)
        return mu, sigma

    def get_action(self, features):
        mu, sigma = self.forward(features)
        dist = torch.distributions.Normal(mu, sigma)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action, log_prob

    def evaluate_actions(self, features, actions):
        mu, sigma = self.forward(features)
        dist = torch.distributions.Normal(mu, sigma)
        log_prob = dist.log_prob(actions).sum(dim=-1)
        entropy = dist.entropy().sum(dim=-1)
        return log_prob, entropy

class Critic(nn.Module):
    def __init__(self, feature_dim=512):
        super().__init__()
        self.value_head = nn.Linear(feature_dim, 1)
        nn.init.orthogonal_(self.value_head.weight, gain=1.0)
        nn.init.zeros_(self.value_head.bias)

    def forward(self, features):
        return self.value_head(features).squeeze(-1)

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn    = CNN()
        self.actor  = Actor(feature_dim=512, action_dim=3)
        self.critic = Critic(feature_dim=512)

    def forward(self, obs):
        return self.cnn(obs)

    def get_action(self, obs):
        features = self.forward(obs)
        action, log_prob = self.actor.get_action(features)
        value = self.critic(features)
        return action, log_prob, value

    def evaluate(self, obs, actions):
        features = self.forward(obs)
        log_prob, entropy = self.actor.evaluate_actions(features, actions)
        value = self.critic(features)
        return log_prob, entropy, value
