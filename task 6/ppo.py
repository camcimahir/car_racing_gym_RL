import torch
import torch.nn.functional as F


class PPO:

    def __init__(self, model, lr=2.5e-4, gamma=0.99,
                 value_coef=0.5, entropy_coef=0.01, max_grad_norm=0.5,
                 clip_eps=0.2, n_epochs=4, mini_batch_size=256):
        self.model          = model
        self.gamma          = gamma
        self.value_coef     = value_coef
        self.entropy_coef   = entropy_coef
        self.max_grad_norm  = max_grad_norm
        self.clip_eps       = clip_eps
        self.n_epochs       = n_epochs
        self.mini_batch_size = mini_batch_size
        self.optimizer      = torch.optim.Adam(model.parameters(), lr=lr)

    def update(self, rollout, device):
        obs        = torch.FloatTensor(rollout['obs']).to(device)
        actions    = torch.FloatTensor(rollout['actions']).to(device)
        returns    = torch.FloatTensor(rollout['returns']).to(device)
        advantages = torch.FloatTensor(rollout['advantages']).to(device)
        old_lp     = torch.FloatTensor(rollout['log_probs']).to(device)

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        batch_size = obs.shape[0]
        sum_actor = sum_critic = sum_entropy = sum_total = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            perm = torch.randperm(batch_size, device=device)

            for start in range(0, batch_size, self.mini_batch_size):
                idx = perm[start : start + self.mini_batch_size]

                mb_obs = obs[idx]
                mb_act = actions[idx]
                mb_ret = returns[idx]
                mb_adv = advantages[idx]
                mb_olp = old_lp[idx]

                new_lp, ent, val = self.model.evaluate(mb_obs, mb_act)

                ratio = (new_lp - mb_olp).exp()
                surr1 = ratio * mb_adv
                surr2 = ratio.clamp(1 - self.clip_eps,
                                    1 + self.clip_eps) * mb_adv
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss  = F.smooth_l1_loss(val, mb_ret)
                entropy_loss = -ent.mean()

                loss = (actor_loss
                        + self.value_coef  * critic_loss
                        + self.entropy_coef * entropy_loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),
                                               self.max_grad_norm)
                self.optimizer.step()

                sum_actor   += actor_loss.item()
                sum_critic  += critic_loss.item()
                sum_entropy += -entropy_loss.item()
                sum_total   += loss.item()
                n_updates   += 1

        return {
            'total_loss':  sum_total  / n_updates,
            'actor_loss':  sum_actor  / n_updates,
            'critic_loss': sum_critic / n_updates,
            'entropy':     sum_entropy / n_updates,
        }
