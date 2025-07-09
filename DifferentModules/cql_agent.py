import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import os

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, *args):
        self.buffer.append(args)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(states),
            torch.FloatTensor(actions),
            torch.FloatTensor(rewards).unsqueeze(1),
            torch.FloatTensor(next_states),
            torch.FloatTensor(dones).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )

    def forward(self, x):
        return self.net(x)


class CQLAgent:
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        self.q1 = MLP(self.state_dim + self.action_dim, 1).to(device)
        self.q2 = MLP(self.state_dim + self.action_dim, 1).to(device)
        self.q1_target = MLP(self.state_dim + self.action_dim, 1).to(device)
        self.q2_target = MLP(self.state_dim + self.action_dim, 1).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy = MLP(self.state_dim, self.action_dim).to(device)

        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.memory = ReplayBuffer()
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.alpha = 5.0  # conservative loss coefficient

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

    def act(self, state, add_noise=False):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action = self.policy(state_tensor).cpu().data.numpy().flatten()
        if add_noise:
            norm_action += 0.1 * np.random.randn(self.action_dim)
        norm_action = np.clip(norm_action, -1, 1)
        return self.denormalize_action(norm_action)

    def remember(self, state, action, reward, next_state, done):
        norm_action = self.normalize_action(action)
        self.memory.append(state, norm_action, reward, next_state, done)

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.memory.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        with torch.no_grad():
            next_actions = self.policy(next_states)
            q1_target = self.q1_target(torch.cat([next_states, next_actions], dim=1))
            q2_target = self.q2_target(torch.cat([next_states, next_actions], dim=1))
            target_q = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        q1_pred = self.q1(torch.cat([states, actions], dim=1))
        q2_pred = self.q2(torch.cat([states, actions], dim=1))
        q1_loss = nn.MSELoss()(q1_pred, target_q)
        q2_loss = nn.MSELoss()(q2_pred, target_q)

        # Conservative Q loss (避免 inplace)
        with torch.no_grad():
            random_actions = torch.FloatTensor(self.batch_size * 10, self.action_dim).uniform_(-1, 1).to(self.device)
            repeated_states = states.unsqueeze(1).repeat(1, 10, 1).reshape(-1, self.state_dim)

        cat_states = torch.cat([repeated_states, random_actions], dim=1)
        q1_vals = self.q1(cat_states)
        q2_vals = self.q2(cat_states)

        conservative_q1 = torch.logsumexp(q1_vals, dim=0).mean() - q1_pred.mean()
        conservative_q2 = torch.logsumexp(q2_vals, dim=0).mean() - q2_pred.mean()
        conservative_loss = self.alpha * (conservative_q1 + conservative_q2)

        total_q1_loss = q1_loss + self.alpha * conservative_q1
        total_q2_loss = q2_loss + self.alpha * conservative_q2

        self.q1_optim.zero_grad()
        total_q1_loss.backward(retain_graph=True)
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        total_q2_loss.backward()
        self.q2_optim.step()

        policy_actions = self.policy(states)
        policy_loss = -self.q1(torch.cat([states, policy_actions], dim=1)).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # Soft update
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "conservative_loss": conservative_loss.item()
        }

    def save(self, path):
        torch.save({
            'q1': self.q1.state_dict(),
            'q2': self.q2.state_dict(),
            'policy': self.policy.state_dict()
        }, path)

    def load(self, path):
        ckpt = torch.load(path)
        self.q1.load_state_dict(ckpt['q1'])
        self.q2.load_state_dict(ckpt['q2'])
        self.policy.load_state_dict(ckpt['policy'])

import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def train_cql(env, agent, episodes=1000, max_steps=500, log_prefix="simple_cql_exp", model_path=None, pretrain_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/cql_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # 日志配置
    logger = logging.getLogger(f"cql_logger_{log_prefix}")
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

    for ep in tqdm(range(episodes), desc="CQL Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state, add_noise=True)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()
            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            if update_info:
                writer.add_scalar("Loss/Q1", update_info["q1_loss"], global_step)
                writer.add_scalar("Loss/Q2", update_info["q2_loss"], global_step)
                writer.add_scalar("Loss/Policy", update_info["policy_loss"], global_step)
                writer.add_scalar("Loss/Conservative", update_info["conservative_loss"], global_step)

                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | "
                    f"Loss: Q1={update_info['q1_loss']:.2f}, Q2={update_info['q2_loss']:.2f}, "
                    f"Policy={update_info['policy_loss']:.2f}, CQL={update_info['conservative_loss']:.2f}"
                )

            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s"
        )

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] 模型权重已保存为 {ckpt_path}")

    agent.save(final_model_path)
    logger.info(f"✅ 最终模型保存为 {final_model_path}")
    writer.close()
    return rewards
