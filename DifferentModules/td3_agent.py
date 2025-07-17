import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections

import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

class ReplayBuffer:
    def __init__(self, max_size=100000):
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, *args):
        self.buffer.append(args)

    def __len__(self):
        return len(self.buffer)

class TD3Actor(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, state):
        return self.net(state)

class TD3Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        sa = torch.cat([state, action], dim=1)
        return self.q1(sa), self.q2(sa)

class TD3Agent:
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        self.actor = TD3Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = TD3Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = TD3Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.memory = ReplayBuffer()
        self.batch_size = 128
        self.gamma = 0.99
        self.tau = 0.005
        self.policy_noise = 0.2
        self.noise_clip = 0.5
        self.policy_delay = 2
        self.total_it = 0

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

    def remember(self, state, action, reward, next_state, done):
        norm_action = self.normalize_action(action)
        self.memory.append(state, norm_action, reward, next_state, done)

    def select_action(self, state, noise_scale=0.1):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action = self.actor(state_tensor).cpu().data.numpy().flatten()
        norm_action += noise_scale * np.random.randn(self.action_dim)
        norm_action = np.clip(norm_action, -1, 1)
        return self.denormalize_action(norm_action)

    def train_step(self):
        if len(self.memory) < self.batch_size:
            return None, None, None

        self.total_it += 1
        batch = random.sample(self.memory.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        q_value = q1.mean().item()

        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([states, self.actor(states)], dim=1)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return (
            actor_loss.item() if actor_loss is not None else None,
            critic_loss.item(),
            q_value
        )


    def save(self, path):
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_weights(self, weights):
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def load(self, path):
        weights = torch.load(path)
        self.load_weights(weights)

    def get_weights(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    # === 添加在 TD3Agent 类内部最后 ===
    def act(self, state, add_noise=False):
        noise = 0.1 if add_noise else 0.0
        return self.select_action(state, noise_scale=noise)

import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging


logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/td3_training.log', filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_td3(env, agent, episodes=1000, max_steps=500, log_prefix="td3_exp", pretrain_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # === 路径设置 ===
    tb_log_dir = f"runs/td3/{log_prefix}"
    saved_model_dir = f"saved_models/td3/{log_prefix}"
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # === 独立日志器 ===
    log_file = f"logs/td3_training_{log_prefix}.log"
    logger_td3 = logging.getLogger(f"td3_logger_{log_prefix}")
    logger_td3.setLevel(logging.INFO)
    logger_td3.propagate = False
    if not logger_td3.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger_td3.addHandler(fh)

    # === 加载预训练模型（可选） ===
    use_pretrain = pretrain_path is not None and os.path.exists(pretrain_path)
    if use_pretrain:
        ckpt = torch.load(pretrain_path)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        print(f"✅ 成功加载预训练权重: {pretrain_path}")
        logger_td3.info(f"成功加载预训练权重: {pretrain_path}")
    else:
        print("🚫 未加载预训练，将从头开始训练。")
        logger_td3.info("未加载预训练，将从头开始训练。")

    # === 主循环 ===
    for ep in tqdm(range(episodes), desc="TD3 Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, float(done))
            actor_loss, critic_loss, q_value = agent.train_step()

            state = next_state
            ep_reward += reward

            writer.add_scalar("Loss/Critic", critic_loss or 0.0, global_step)
            if actor_loss is not None:
                writer.add_scalar("Loss/Actor", actor_loss, global_step)
            if q_value is not None:
                writer.add_scalar("Q_value", q_value, global_step)

            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info["weight_error"], global_step)
            writer.add_scalar("Metric/Time", info["total_time"], global_step)
            writer.add_scalar("Metric/SlowWeight", info["action"].get("slow_weight", 0.0), global_step)

            global_step += 1
            average_step += 1
            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        actor_loss_str = f"{actor_loss:.4f}" if actor_loss is not None else "-"
        critic_loss_str = f"{critic_loss:.4f}" if critic_loss is not None else "-"

        logger_td3.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info['weight_error']:.2f}g | Time: {info['total_time']:.2f}s | "
            f"SlowWeight: {info['action'].get('slow_weight', 0.0):.2f}g | "
            f"ActorLoss: {actor_loss_str} | CriticLoss: {critic_loss_str}"
        )

        # 保存中间模型
        if ep % 20 == 0 or ep == episodes - 1:
            model_path = os.path.join(saved_model_dir, f"td3_{log_prefix}_ep{ep:04d}.pth")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, model_path)
            logger_td3.info(f"[Episode {ep}] 模型已保存至: {model_path}")

    # ✅ 最终保存
    final_path = os.path.join(saved_model_dir, f"td3_{log_prefix}_final.pth")
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, final_path)
    logger_td3.info(f"✅ 训练完成，最终模型已保存至: {final_path}")

    writer.close()
    return rewards
