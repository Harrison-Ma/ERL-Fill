import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import collections
import random


class RolloutBuffer:
    def __init__(self):
        self.states = []
        self.actions = []
        self.log_probs = []
        self.rewards = []
        self.dones = []
        self.values = []
        self.constraints = []

    def clear(self):
        self.__init__()


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        self.actor_log_std = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        self.constraint_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action_distribution(self, state):
        mean = self.actor_mean(state)
        log_std = self.actor_log_std(state).clamp(-20, 2)
        std = torch.exp(log_std)
        return torch.distributions.Normal(mean, std)

    def act(self, state, deterministic=False):
        dist = self.get_action_distribution(state)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        action = torch.tanh(action)
        return action

    def evaluate(self, state):
        value = self.critic(state)
        constraint_value = self.constraint_critic(state)
        return value, constraint_value


class PPOLagrangian:
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        self.gamma = 0.99
        self.lam = 0.95
        self.eps_clip = 0.2
        self.lr = 3e-4
        self.update_epochs = 10
        self.batch_size = 64
        self.constraint_limit = 0.01
        self.lambda_lr = 0.05

        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)
        self.rollout_buffer = RolloutBuffer()

        self.lagrange_multiplier = torch.tensor(0.0, requires_grad=False, device=device)

        # ✅ 添加 memory 接口，兼容 env 中的 agent.memory 判定
        self.memory = self

    def __len__(self):
        # ✅ 返回 buffer 当前长度，用于 memory 判断
        return len(self.rollout_buffer.states)

    def normalize_action(self, action):
        return np.array([
            2 * (action[i] - low) / (high - low) - 1
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def denormalize_action(self, norm_action):
        return np.array([
            0.5 * (norm_action[i] + 1) * (high - low) + low
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def act(self, state, deterministic=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.act(state_tensor, deterministic)
        return self.denormalize_action(action.detach().cpu().numpy().flatten())

    def remember(self, state, action, reward, next_state, done, cost):
        norm_action = self.normalize_action(action)
        self.rollout_buffer.states.append(torch.FloatTensor(state))
        self.rollout_buffer.actions.append(torch.FloatTensor(norm_action))
        self.rollout_buffer.rewards.append(torch.FloatTensor([reward]))
        self.rollout_buffer.dones.append(torch.FloatTensor([done]))
        value, constraint = self.policy.evaluate(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        self.rollout_buffer.values.append(value.detach().cpu().squeeze(0))
        self.rollout_buffer.constraints.append(torch.FloatTensor([cost]))

    def update(self):
        states = torch.stack(self.rollout_buffer.states).to(self.device)
        actions = torch.stack(self.rollout_buffer.actions).to(self.device)
        rewards = torch.stack(self.rollout_buffer.rewards).to(self.device)
        dones = torch.stack(self.rollout_buffer.dones).to(self.device)
        values = torch.stack(self.rollout_buffer.values).to(self.device)
        constraints = torch.stack(self.rollout_buffer.constraints).to(self.device)

        returns = []
        advantages = []
        constraint_returns = []
        gae = 0
        constraint_gae = 0
        next_value = 0
        next_constraint = 0

        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae + values[i])
            next_value = values[i]

            constraint_delta = constraints[i] + self.gamma * next_constraint * (1 - dones[i]) - constraints[i]
            constraint_gae = constraint_delta + self.gamma * self.lam * (1 - dones[i]) * constraint_gae
            constraint_returns.insert(0, constraint_gae + constraints[i])
            next_constraint = constraints[i]

        advantages = torch.stack(advantages)
        constraint_returns = torch.stack(constraint_returns)

        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(rewards))
            for start in range(0, len(rewards), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_const = constraint_returns[batch_indices]

                curr_values, curr_constraints = self.policy.evaluate(batch_states)
                dist = self.policy.get_action_distribution(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(axis=-1)

                ratio = torch.exp(log_probs - log_probs.detach())
                surrogate1 = ratio * batch_adv
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv

                loss_clip = -torch.min(surrogate1, surrogate2).mean()
                value_loss = nn.MSELoss()(curr_values.squeeze(), batch_adv.squeeze())
                constraint_loss = nn.MSELoss()(curr_constraints.squeeze(), batch_const.squeeze())

                lagrangian_term = self.lagrange_multiplier * constraint_loss
                total_loss = loss_clip + 0.5 * value_loss + lagrangian_term

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        mean_constraint = torch.mean(constraint_returns).item()
        self.lagrange_multiplier += self.lambda_lr * (mean_constraint - self.constraint_limit)
        self.lagrange_multiplier = torch.clamp(self.lagrange_multiplier, 0.0, 10.0)

        self.rollout_buffer.clear()
        return {
            'policy_loss': loss_clip.item(),
            'value_loss': value_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'lambda': self.lagrange_multiplier.item(),
            'entropy': dist.entropy().mean().item()
        }

    def save(self, path):
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier
        }, path)

    def load(self, path):
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.lagrange_multiplier = ckpt.get('lagrange_multiplier', torch.tensor(0.0)).to(self.device)


import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_ppo_lagrangian(env, agent, episodes=1000, max_steps=500, log_prefix="ppo_lagrangian_exp", model_path=None, pretrain_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/ppo_lagrangian_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # 日志配置
    logger = logging.getLogger(f"ppo_lagrangian_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    if pretrain_path and os.path.exists(pretrain_path):
        agent.load(pretrain_path)
        print(f"✅ 成功加载预训练模型: {pretrain_path}")
        logger.info(f"成功加载预训练模型: {pretrain_path}")
    else:
        print("❌ 未加载预训练模型，从头开始训练")
        logger.info("未加载预训练模型，从头开始训练")

    for ep in tqdm(range(episodes), desc="PPO-Lagrangian Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)

            cost = info.get("constraint_cost", 0.0)
            agent.remember(state, action, reward, next_state, done, cost)

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # tensorboard scalar
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Cost/Step", cost, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)

            if done:
                break

        update_info = agent.update()

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s"
        )

        if update_info:
            logger.info(
                f"λ: {update_info.get('lambda', '-')} | PolicyLoss: {update_info.get('policy_loss', '-'):.2f} | "
                f"ValueLoss: {update_info.get('value_loss', '-'):.2f} | Entropy: {update_info.get('entropy', '-'):.2f}"
            )
            writer.add_scalar("Loss/Policy", update_info.get('policy_loss', 0.0), ep)
            writer.add_scalar("Loss/Value", update_info.get('value_loss', 0.0), ep)
            writer.add_scalar("Loss/Entropy", update_info.get('entropy', 0.0), ep)
            writer.add_scalar("Lambda", update_info.get('lambda', 0.0), ep)

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] 模型权重已保存为 {ckpt_path}")

    agent.save(final_model_path)
    logger.info(f"✅ 最终模型保存为 {final_model_path}")
    writer.close()
    return rewards
