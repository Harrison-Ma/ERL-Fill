# import SysOut
from CommonInterface.modbus_slave import modbus_slave_client
from VirtualWeightController import VirtualWeightController

from torch.utils.tensorboard import SummaryWriter

import torch.nn.functional as F
from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np
import random
import time
from pymodbus.client import ModbusSerialClient

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

from tqdm import tqdm  # ✅ 添加进度条支持

import platform

import logging
import os

from CommonInterface.Logger import init_logger

from EmotionModule import EmotionModuleNone


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, param_bounds, emotion_dim=3, hidden_size=128, dropout_rate=0.2):
        super(Actor, self).__init__()
        self.dropout_rate = dropout_rate

        # 输入层（含情感维度）
        self.input_layer = nn.Linear(state_dim + emotion_dim, hidden_size)

        # 深层网络结构
        self.hidden = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Softsign()  # 输出控制在 (-1, 1)
        )

        # 情感扰动权重
        self.emotion_weights = nn.Parameter(torch.randn(emotion_dim, action_dim) * 0.2)

        # 参数缩放和偏移（动作映射）
        self._init_param_scaling(param_bounds)

    def _init_param_scaling(self, param_bounds):
        scale_params, offset_params = [], []
        for key in param_bounds.keys():
            low, high = param_bounds[key]
            scale = (high - low) / 2.0
            offset = (high + low) / 2.0
            scale_params.append(scale)
            offset_params.append(offset)
        self.register_buffer('scale_params', torch.tensor(scale_params, dtype=torch.float32))
        self.register_buffer('offset_params', torch.tensor(offset_params, dtype=torch.float32))

    def forward(self, state, emotion):
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if emotion.size(0) != state.size(0):
            emotion = emotion.expand(state.size(0), -1)

        x = torch.cat([state, emotion], dim=1)
        x = self.input_layer(x)
        base_action = self.hidden(x)  # [-1, 1]

        # === 情绪因子解构 ===
        curiosity = emotion[:, 0:1]      # 探索
        conserv   = emotion[:, 1:2]      # 保守
        anxiety   = emotion[:, 2:3]      # 焦虑

        # === 原始扰动计算 ===
        raw_effect = torch.tanh((emotion @ self.emotion_weights) * 1.5)  # [-1, 1]

        # === 计算 alpha 系数 ===
        alpha = 0.1 + 0.9 * anxiety                  # 基础随焦虑增强
        alpha *= (1.0 - 0.5 * conserv)               # 保守性削弱幅度
        alpha *= (1.0 + 0.5 * curiosity)             # 探索增强调整弹性
        alpha = torch.clamp(alpha, min=0.01, max=1)  # 限制扰动范围

        # === 调整扰动方向（鼓励逃离饱和区） ===
        modulator = (1 - torch.abs(base_action)) ** 2
        sign_flip = -torch.sign(base_action)  # 如果在边界附近，就向中心反向扰动
        adjusted_emotion_effect = alpha * raw_effect * sign_flip * modulator  # [-α, α]

        # === 最终动作计算 ===
        raw_action = base_action + adjusted_emotion_effect
        scale_params = self.scale_params.to(raw_action.device)
        offset_params = self.offset_params.to(raw_action.device)
        scaled_action = raw_action * scale_params + offset_params

        return scaled_action


# class Critic(nn.Module):
#     def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=128):
#         super(Critic, self).__init__()
#         # 输入层增加情感维度
#         self.net = nn.Sequential(
#             nn.Linear(state_dim + action_dim + emotion_dim, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, hidden_size),
#             nn.ReLU(),
#             nn.Linear(hidden_size, 1)
#         )
#
#     def forward(self, state, action, emotion):
#         return self.net(torch.cat([state, action, emotion], 1))

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256, dropout_rate=0.2):
        super(Critic, self).__init__()
        input_dim = state_dim + action_dim + emotion_dim

        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, 1)
        )

    def forward(self, state, action, emotion):
        # 情感标准化：抑制情感漂移带来的 Q 估计不稳定
        emotion = (emotion - emotion.mean(dim=0, keepdim=True)) / (emotion.std(dim=0, keepdim=True) + 1e-6)

        x = torch.cat([state, action, emotion], dim=1)
        q = self.net(x)
        return torch.clamp(q, -1e6, 1e6)

class PrioritizedReplayBuffer:
    def __init__(self, capacity=100000, alpha=0.6, epsilon=1e-4):
        self.capacity = capacity
        self.alpha = alpha
        self.epsilon = epsilon
        self.buffer = []
        self.priorities = []

    def add(self, transition, priority):
        """添加经验并计算对应的优先级"""
        p = (abs(priority) + self.epsilon) ** self.alpha
        p = np.nan_to_num(p, nan=self.epsilon)  # ✅ 防止出现 NaN

        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            idx = np.argmin(self.priorities)  # ✅ 替换优先级最低的
            self.buffer[idx] = transition
            self.priorities[idx] = p

    def sample(self, batch_size):
        """根据概率分布采样"""
        priorities = np.array(self.priorities, dtype=np.float32)

        # ✅ 防止 priorities.sum() 为 0 或 NaN
        total = priorities.sum()
        if total <= 0 or np.isnan(total):
            priorities += self.epsilon
            total = priorities.sum()

        probs = priorities / total
        probs = np.nan_to_num(probs, nan=1.0 / len(probs))  # ✅ 防止出现 NaN

        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]

    def __len__(self):
        return len(self.buffer)

class DDPGAgent:
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu', use_td_error=False):
        param_bounds = env.bounds
        self.env = env
        self.device = device
        self.use_td_error = use_td_error

        self.emotion = EmotionModuleNone()

        self.actor = Actor(env.state_dim, env.action_dim, param_bounds).to(device)
        self.actor_target = Actor(env.state_dim, env.action_dim, param_bounds).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = Critic(env.state_dim, env.action_dim).to(device)
        self.critic_target = Critic(env.state_dim, env.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 128
        self.memory_size = 100000
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        self.train_step = 1  # 或 self.episode_count，如果以轮次为基础
        self.lr_decay_rate = 0.999  # 每步或每轮衰减 0.5%

        self.target_update_freq = 1  # 每 2 步更新一次
        self.learn_step = 0

        self.actor_update_freq = 1  # 每隔 2 次 Critic 学习更新一次 Actor

        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # self.memory = []  # 切换为列表以支持TD-error存储
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_size)
        self.ou_noise = OUNoise(env.action_dim)

    def act(self, state, add_noise=True):
        emotion_state = self.emotion.get_emotion()

        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        emotion_tensor = torch.from_numpy(emotion_state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_action = self.actor(state_tensor, emotion_tensor)

        if not add_noise:
            return scaled_action.cpu().numpy().flatten()

        noise = self.ou_noise.sample()
        scale = self.actor.scale_params.cpu().numpy()
        offset = self.actor.offset_params.cpu().numpy()

        final_action = np.clip(
            scaled_action.cpu().numpy().flatten() + noise * scale,
            offset - scale,
            offset + scale
        )
        return final_action

    def remember(self, state, action, reward, next_state, done):
        emotion_state = self.emotion.get_emotion()

        action = np.array(action)
        norm_action = (action - self.actor.offset_params.cpu().numpy()) / self.actor.scale_params.cpu().numpy()
        norm_action = np.clip(norm_action, -1.0, 1.0)

        td_error = 0
        if self.use_td_error:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)
                emotion_tensor = torch.FloatTensor(emotion_state).unsqueeze(0).to(self.device)

                next_emotion_tensor = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)
                next_action = self.actor_target(next_state_tensor, next_emotion_tensor)
                target_q = self.critic_target(next_state_tensor, next_action, next_emotion_tensor)
                expected_q = reward + (1 - int(done)) * self.gamma * target_q.item()
                current_q = self.critic(state_tensor, action_tensor, emotion_tensor).item()
                td_error = abs(expected_q - current_q)

        # self.memory.append((state, norm_action, reward, next_state, done, emotion_state, td_error))

        # if len(self.memory) > self.memory_size:
        #     self.memory.pop(0)
        self.memory.add(
            transition=(state, norm_action, reward, next_state, done, emotion_state, td_error),
            priority=td_error  # 使用 TD-error 初始化优先级
        )

    def compute_dynamic_lr(emotion, base_lr, role="critic", min_lr=5e-5, max_lr=2e-3):
        anxiety, conservativeness, curiosity = emotion

        if role == "critic":
            factor = 1.0 - 0.5 * anxiety + 0.4 * conservativeness + 0.1 * curiosity
        elif role == "actor":
            factor = 1.0 + 0.4 * curiosity - 0.3 * conservativeness
        else:
            factor = 1.0

        return float(np.clip(base_lr * factor, min_lr, max_lr))

    def learn(self):
        if len(self.memory) < self.batch_size:
            return

        # ----- 1. 采样策略 -----
        # if self.use_td_error:
        #     td_errors = np.array([m[6] for m in self.memory]) + 1e-6
        #     prob = td_errors / td_errors.sum()
        #     indices = np.random.choice(len(self.memory), self.batch_size, p=prob)
        #     batch = [self.memory[i] for i in indices]
        # else:
        #     batch = random.sample(self.memory, self.batch_size)
        if self.use_td_error:
            batch = self.memory.sample(self.batch_size)
        else:
            batch = random.sample(self.memory.buffer, self.batch_size)

        # ----- 2. 解包 & Tensor 转换 -----
        states, actions, rewards, next_states, dones, emotions, _ = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        emotions = torch.FloatTensor(emotions).to(self.device)

        # ✅ reward 缩放
        reward_scale = 0.01
        rewards = rewards * reward_scale

        # ✅ 获取当前平均情感指标
        anxiety = torch.mean(emotions[:, 2]).item()
        conservativeness = torch.mean(emotions[:, 1]).item()
        curiosity = torch.mean(emotions[:, 0]).item()

        actor_factor = 1.0 + 0.8 * curiosity - 0.4 * conservativeness + 0.2 * anxiety
        critic_factor = 1.0 - 0.3 * anxiety + 0.6 * conservativeness
        # ✅ 融合情感调节机制，动态调整学习率
        # critic_factor = 1.0 - 0.5 * anxiety + 0.4 * conservativeness + 0.1 * curiosity
        # actor_factor = 1.0 + 0.4 * curiosity - 0.3 * conservativeness

        # critic_scaled_lr = float(np.clip(self.critic_lr * critic_factor, self.critic_lr*0.5, self.critic_lr*5))
        # actor_scaled_lr = float(np.clip(self.actor_lr * actor_factor, self.actor_lr*0.5, self.actor_lr*5))

        # ✅ 加入 step 衰减项（每次学习逐步衰减）
        decay_factor = self.lr_decay_rate ** self.train_step  # 越往后越小

        critic_scaled_lr = float(np.clip(self.critic_lr * critic_factor * decay_factor,
                                         self.critic_lr * 0.1, self.critic_lr * 3))
        actor_scaled_lr = float(np.clip(self.actor_lr * actor_factor * decay_factor,
                                        self.actor_lr * 0.1, self.actor_lr * 3))

        # print("critic_scaled_lr:",critic_scaled_lr)
        # ✅ 应用到优化器
        for g in self.critic_optim.param_groups:
            g["lr"] = critic_scaled_lr
        for g in self.actor_optim.param_groups:
            g["lr"] = actor_scaled_lr

        # ----- 3. Critic 更新 -----
        with torch.no_grad():
            next_emotion_tensor = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)
            next_emotion_tensor = next_emotion_tensor.expand(next_states.size(0), -1)
            next_actions = self.actor_target(next_states, next_emotion_tensor)
            target_q = self.critic_target(next_states, next_actions, next_emotion_tensor)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions, emotions)
        critic_loss = nn.SmoothL1Loss()(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optim.step()

        # ----- 4. Actor 更新 -----
        actor_loss = None  # 先占位

        if self.learn_step % self.actor_update_freq == 0:
            action_out = self.actor(states, emotions)

            # 🔥 原始 loss：最大化 Q 值
            actor_loss = -self.critic(states, action_out, emotions).mean()

            # ✅ 多样性正则项（鼓励 std 越大越好）
            diversity_term = torch.std(action_out, dim=1).mean()
            actor_loss -= 0.03 * diversity_term  # 动作多样性激励

            # ✅ L2 正则项（鼓励权重不要过大）
            l2_lambda = 1e-4  # 控制正则强度（可调）
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.actor.parameters():
                l2_reg += torch.norm(param, p=2)
            actor_loss += l2_lambda * l2_reg  # 加到最终 loss 中

            # ✅ 反向传播 + 梯度裁剪
            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_optim.step()

        # ----- 5. 软更新目标网络 -----
        self.learn_step += 1  # ✅ 步数递增
        # self.train_step += 1
        #
        # print("self.train_step:",self.train_step)

        if self.learn_step % self.target_update_freq == 0:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # logger.debug(
        #     f"[Anxiety: {anxiety:.2f}] Critic LR: {critic_scaled_lr:.6f}, Actor LR: {actor_scaled_lr:.6f} | "
        #     f"Losses - Critic: {getattr(critic_loss, 'item', lambda: 'None')()}, "
        #     f"Actor: {getattr(actor_loss, 'item', lambda: 'None')()}"
        # )

        # self.memory.add(transition, new_priority)
        return {"critic_loss": critic_loss.item() if critic_loss is not None else None,
                "actor_loss": actor_loss.item() if actor_loss is not None else None}

    def load_weights(self, weights):
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def get_weights(self):
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    # def save(self, filename):
    #     torch.save({
    #         'actor': self.actor.state_dict(),
    #         'critic': self.critic.state_dict(),
    #         'emotion': self.emotion.current_emotion,
    #         'emotion_transformer': self.emotion.transformer.state_dict()  # ✅ 新增部分
    #     }, filename)

    def save(self, path):
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'train_step': self.train_step,
        }

        # ✅ 可选保存 emotion transformer
        if hasattr(self, "emotion") and hasattr(self.emotion, "transformer"):
            save_dict['emotion_transformer'] = self.emotion.transformer.state_dict()

        torch.save(save_dict, path)
        # logger.info(f"模型已保存至 {path}")

    def load(self, filename):
        checkpoint = torch.load(filename, map_location=self.device,weights_only=False)
        # checkpoint = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        # ✅ 安全加载 current_emotion
        if 'emotion' in checkpoint:
            self.emotion.current_emotion = checkpoint['emotion']
        else:
            print("⚠️ 未发现 emotion 状态字段，跳过 current_emotion 加载")

        # ✅ 安全加载 EmotionTransformer（如果存在）
        if 'emotion_transformer' in checkpoint:
            if hasattr(self.emotion, 'transformer'):
                self.emotion.transformer.load_state_dict(checkpoint['emotion_transformer'])
                print("✅ EmotionTransformer 权重已成功加载")
            else:
                print("⚠️ 检查点包含 EmotionTransformer 权重，但当前 emotion 模块不支持 transformer，已跳过加载")
        else:
            print("⚠️ 未发现 EmotionTransformer 权重字段，跳过加载")

import shutil  # 确保导入

def reset_actor_scaling(agent, new_bounds):
    """
    重置 Actor 和 Actor-Target 的参数缩放信息（scale_params 和 offset_params）

    参数:
        agent: DDPGAgent 实例
        new_bounds: dict, 例如 env.bounds
    """
    if hasattr(agent.actor, '_init_param_scaling'):
        agent.actor._init_param_scaling(new_bounds)
    if hasattr(agent.actor_target, '_init_param_scaling'):
        agent.actor_target._init_param_scaling(new_bounds)
    print("✅ Actor 缩放参数已根据新 bounds 重新初始化。")

def train_ddpg(env, agent, episodes=1000, max_steps=500, log_prefix="ddpg_emotion", logger=None):
    if logger is None:
        logger = init_logger(log_prefix=log_prefix)
    rewards = []
    env.max_steps = max_steps

    tb_log_dir = f"runs/{log_prefix}"
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    global_step = 0
    pretrain_path = "pre_train_pth/AIM_v1_sim_24900_25000.pth"
    use_pretrain = 0
    if not use_pretrain:
        logger.info("已选择不加载预训练权重，将从头开始训练。")
        print("已选择不加载预训练权重，将从头开始训练。")
    elif os.path.exists(pretrain_path):
        weights = torch.load(pretrain_path, weights_only=False)
        # weights = torch.load(pretrain_path)
        agent.load_weights(weights)
        reset_actor_scaling(agent, env.bounds)
        logger.info(f"成功加载预训练权重: {pretrain_path}")
        print(f"成功加载预训练权重: {pretrain_path}")
    else:
        logger.info("未发现预训练权重，将从头开始训练。")
        print("未发现预训练权重，将从头开始训练。")

    start_100 = time.time()  # ⏱️ 初始化100回合计时器

    for ep in tqdm(range(episodes), desc="Training Progress", ncols=100):
        state = env.reset()
        episode_reward = 0
        agent.ou_noise.reset()
        agent.ou_noise.anneal()
        average_step = 0

        for step in range(max_steps):
            action = np.transpose(agent.act(state))
            next_state, reward, done, info = env.step(action)
            movement = next_state - state
            agent.emotion.update(reward, movement)
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            episode_reward += reward

            loss = agent.learn()
            if loss:
                if loss.get('actor_loss') is not None:
                    writer.add_scalar("Loss/Actor", loss['actor_loss'], global_step)
                if loss.get('critic_loss') is not None:
                    writer.add_scalar("Loss/Critic", loss['critic_loss'], global_step)
                writer.add_scalar("LR/Actor", agent.actor_optim.param_groups[0]['lr'], global_step)
                writer.add_scalar("LR/Critic", agent.critic_optim.param_groups[0]['lr'], global_step)

            cur_emotion = agent.emotion.get_emotion()
            if isinstance(cur_emotion, torch.Tensor):
                cur_emotion = cur_emotion.detach().cpu().numpy().squeeze()
            cur_emotion = np.array(cur_emotion).flatten()
            writer.add_scalar("Emotion/Curiosity", float(cur_emotion[0]), global_step)
            writer.add_scalar("Emotion/Conservativeness", float(cur_emotion[1]), global_step)
            writer.add_scalar("Emotion/Anxiety", float(cur_emotion[2]), global_step)

            if step % (max_steps // 10) == 0 or step == max_steps - 1:
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | Reward: {reward:.2f} | Loss: {loss} | Emotion: {np.round(cur_emotion, 2)}"
                )

            global_step += 1
            average_step += 1
            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        rewards.append(episode_reward)
        writer.add_scalar("Reward/Episode", episode_reward, ep)
        writer.add_scalar("Reward/Episode(average)", episode_reward / average_step, ep)
        writer.add_scalar("Metric/Weight", info["final_weight"], ep)
        writer.add_scalar("Metric/Error", info["weight_error"], ep)
        writer.add_scalar("Metric/Time", info["total_time"], ep)
        writer.add_scalar("Reward/Step", reward, ep)

        for key, value in info["action"].items():
            writer.add_scalar(f"Action/{key}", value, ep)

        writer.add_scalar("Metric/SlowWeight", info["action"].get("slow_weight", 0.0), ep)

        if ep % 1 == 0 or ep == episodes - 1:
            cur_emotion = agent.emotion.get_emotion()
            if isinstance(cur_emotion, torch.Tensor):
                cur_emotion = cur_emotion.detach().cpu().numpy().squeeze()
            cur_emotion = np.array(cur_emotion).flatten()

            actor_loss = f"{loss.get('actor_loss'):.4f}" if loss and loss.get("actor_loss") is not None else "-"
            critic_loss = f"{loss.get('critic_loss'):.4f}" if loss and loss.get("critic_loss") is not None else "-"
            slow_weight = info["action"].get("slow_weight", 0.0)

            logger.info(
                f"[Episode {ep:04d}] "
                f"TotalReward: {episode_reward:.2f} | "
                f"AvgReward: {episode_reward / average_step:.2f} | "
                f"WeightErr: {info['weight_error']:.2f}g | "
                f"Time: {info['total_time']:.2f}s | "
                f"SlowWeight: {slow_weight:.2f}g | "
                f"Emotion: [Curi: {cur_emotion[0]:.2f}, Cons: {cur_emotion[1]:.2f}, Anx: {cur_emotion[2]:.2f}] | "
                f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
            )

        if ep > 0 and ep % 100 == 0:
            duration_100 = time.time() - start_100
            logger.info(f"⏱️ 第 {ep-100}~{ep} 回合共耗时 {duration_100:.2f} 秒，平均 {duration_100 / 100:.2f} 秒/回合")
            start_100 = time.time()

        if ep % 10 == 0 or ep == episodes - 1:
            torch.save(agent.get_weights(), "ddpg_emotion_agent.pth")
            logger.info(f"[Episode {ep}] 模型权重已保存为 ddpg_emotion_agent.pth")

        agent.train_step += 1

    writer.close()
    return rewards


# # Ornstein-Uhlenbeck噪声生成器（保持不变）
# class OUNoise:
#     def __init__(self, size, mu=0.0, theta=0.15, sigma=0.2):
#         self.mu = mu * np.ones(size)
#         self.theta = theta
#         self.sigma = sigma
#         self.reset()
#
#     def reset(self):
#         self.state = np.copy(self.mu)
#
#     def sample(self):
#         dx = self.theta * (self.mu - self.state)
#         dx += self.sigma * np.random.randn(len(self.state))
#         self.state += dx
#         return self.state

class OUNoise:
    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.6, min_sigma=0.1, decay=0.9995):
        self.size = size
        self.mu = mu * np.ones(size, dtype=np.float32)
        self.theta = theta
        self.sigma = sigma
        self.min_sigma = min_sigma
        self.decay = decay
        self.reset()

    def reset(self):
        self.state = np.copy(self.mu)

    def sample(self):
        dx = self.theta * (self.mu - self.state)
        dx += self.sigma * np.random.randn(self.size).astype(np.float32)
        self.state += dx
        return self.state

    def anneal(self):
        self.sigma = max(self.min_sigma, self.sigma * self.decay)