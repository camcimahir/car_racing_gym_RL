import os
import time
import numpy as np
import torch
import gymnasium as gym
from torch.utils.tensorboard import SummaryWriter

from model import ActorCritic
from ppo import PPO
from utils import FrameStack, to_tensor, compute_gae

# ── Hyperparameters ────────────────────────────────────────────────────────────
CONFIG = {
    'env_name':      'CarRacing-v3',
    'seed':          42,

    'num_envs':      16,
    'total_steps':   2_000_000,
    'n_steps':       256,
    'gamma':         0.99,
    'gae_lambda':    0.95,

    'lr':            2.5e-4,
    'value_coef':    0.5,
    'entropy_coef':  0.01,
    'max_grad_norm': 0.5,
    'clip_eps':      0.2,
    'ppo_epochs':    4,
    'mini_batch_size': 256,

    'log_interval':  1,
    'save_interval': 50,
    'save_dir':      os.path.expanduser('~/persistent/models/ppo_carracing'),
    'log_dir':       os.path.expanduser('~/persistent/logs/ppo_carracing'),
}
# ──────────────────────────────────────────────────────────────────────────────
def collect_rollouts(envs, frame_stacks, model, n_steps, device, obs, current_ep_rewards):

    num_envs = CONFIG['num_envs']

    # Storage arrays for batching
    obs_batch = np.zeros((n_steps, num_envs, 4, 96, 96), dtype=np.float32)
    actions_batch = np.zeros((n_steps, num_envs, 3), dtype=np.float32)
    rewards_batch = np.zeros((n_steps, num_envs), dtype=np.float32)
    dones_batch = np.zeros((n_steps, num_envs), dtype=np.float32)
    values_batch = np.zeros((n_steps, num_envs), dtype=np.float32)

    finished_episodes = []

    for step in range(n_steps):
        obs_tensor = to_tensor(obs, device)

        with torch.no_grad():
            action_idx, log_prob, value = model.get_action(obs_tensor)

        # Steering: [-1.0, 1.0], Gas: [0.0, 1.0], Brake: [0.0, 1.0]
        actions_np = action_idx.cpu().numpy()
        actions_np[:, 0] = np.clip(actions_np[:, 0], -1.0, 1.0)
        actions_np[:, 1] = np.clip(actions_np[:, 1], 0.0, 1.0)
        actions_np[:, 2] = np.clip(actions_np[:, 2], 0.0, 1.0)

        next_raw_obs, rewards, terminateds, truncateds, infos = envs.step(actions_np)
        dones = terminateds | truncateds

        obs_batch[step] = obs
        actions_batch[step] = actions_np  # Store the continuous actions
        rewards_batch[step] = rewards
        dones_batch[step] = dones
        values_batch[step] = value.cpu().numpy()

        for i in range(num_envs):
            current_ep_rewards[i] += rewards[i]

            if dones[i]:
                finished_episodes.append(current_ep_rewards[i])
                current_ep_rewards[i] = 0.0
                obs[i] = frame_stacks[i].reset(next_raw_obs[i])
            else:
                obs[i] = frame_stacks[i].step(next_raw_obs[i])

    last_obs_tensor = to_tensor(obs, device)
    with torch.no_grad():
        _, _, last_values = model.get_action(last_obs_tensor)
    last_values = last_values.cpu().numpy()

    advantages_batch = np.zeros_like(rewards_batch)
    returns_batch    = np.zeros_like(rewards_batch)
    for i in range(num_envs):
        advantages_batch[:, i], returns_batch[:, i] = compute_gae(
            rewards_batch[:, i],
            values_batch[:, i],
            dones_batch[:, i],
            last_values[i],
            CONFIG['gamma'],
            CONFIG['gae_lambda'],
        )

    obs_flat = obs_batch.reshape(-1, 4, 96, 96)
    act_flat = actions_batch.reshape(-1, 3)

    with torch.no_grad():
        old_lp, _, _ = model.evaluate(
            torch.FloatTensor(obs_flat).to(device),
            torch.FloatTensor(act_flat).to(device),
        )

    rollout = {
        'obs':        obs_flat,
        'actions':    act_flat,
        'values':     values_batch.flatten(),
        'returns':    returns_batch.flatten(),
        'advantages': advantages_batch.flatten(),
        'log_probs':  old_lp.cpu().numpy(),
    }

    return rollout, finished_episodes, obs, current_ep_rewards


def save_checkpoint(model, optimizer_state, update, total_steps, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    path = os.path.join(save_dir, f'checkpoint_{update}.pt')
    torch.save({
        'update':      update,
        'total_steps': total_steps,
        'model_state': model.state_dict(),
        'optimizer':   optimizer_state,
    }, path)


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    def make_env():
        return gym.make(CONFIG['env_name'], render_mode=None)
    
    envs = gym.vector.AsyncVectorEnv([make_env for _ in range(CONFIG['num_envs'])])
    
=    frame_stacks = [FrameStack(n=4) for _ in range(CONFIG['num_envs'])]

    model = ActorCritic().to(device)
    agent = PPO(
        model=model,
        lr=CONFIG['lr'],
        gamma=CONFIG['gamma'],
        value_coef=CONFIG['value_coef'],
        entropy_coef=CONFIG['entropy_coef'],
        max_grad_norm=CONFIG['max_grad_norm'],
        clip_eps=CONFIG['clip_eps'],
        n_epochs=CONFIG['ppo_epochs'],
        mini_batch_size=CONFIG['mini_batch_size'],
    )

    writer = SummaryWriter(log_dir=CONFIG['log_dir'])
    
    total_steps = 0
    update = 0
    all_ep_rewards = []
    start_time = time.time()

    raw_obs, _ = envs.reset()
    obs = np.array([fs.reset(raw_obs[i]) for i, fs in enumerate(frame_stacks)])
    current_ep_rewards = np.zeros(CONFIG['num_envs'])

    print("Starting vectorized training...\n")

    while total_steps < CONFIG['total_steps']:
        rollout, ep_rewards, obs, current_ep_rewards = collect_rollouts(
            envs, frame_stacks, model, CONFIG['n_steps'], device, obs, current_ep_rewards
        )
        
        total_steps += CONFIG['n_steps'] * CONFIG['num_envs']
        update += 1

        all_ep_rewards.extend(ep_rewards)
        losses = agent.update(rollout, device)

        if update % CONFIG['log_interval'] == 0:
            fps = total_steps / (time.time() - start_time)
            recent = all_ep_rewards[-100:] if all_ep_rewards else [0]
            mean_reward = np.mean(recent)

            writer.add_scalar('Loss/total',   losses['total_loss'],  total_steps)
            writer.add_scalar('Loss/actor',   losses['actor_loss'],  total_steps)
            writer.add_scalar('Loss/critic',  losses['critic_loss'], total_steps)
            writer.add_scalar('Reward/mean_last_100', mean_reward, total_steps)
            writer.add_scalar('Performance/fps', fps, total_steps)

            print(f"Update {update:4d} | Steps {total_steps:9,} | FPS {fps:5.0f} | Mean Reward: {mean_reward:6.1f} | Loss: {losses['total_loss']:.4f} | Entropy: {losses['entropy']:.3f}")

        if update % CONFIG['save_interval'] == 0:
            save_checkpoint(model, agent.optimizer.state_dict(), update, total_steps, CONFIG['save_dir'])

    envs.close()
    print("\nTraining complete!")

if __name__ == '__main__':
    train()
