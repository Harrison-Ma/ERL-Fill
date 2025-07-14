import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque


class GaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = self.net(state)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state):
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        return self.q(torch.cat([state, action], dim=-1))


import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

from EmotionModule import EmotionModuleNone


class SACAgent:
    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.gamma = 0.99
        self.tau = 0.005
        self.batch_size = 128
        self.alpha = 0.3  # ✅ 固定 alpha 值

        self.policy = GaussianPolicy(env.state_dim, env.action_dim).to(device)
        self.q1 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q1_target = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2_target = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

        self.memory = type('DummyMemory', (), {
            'buffer': deque(maxlen=100000),
            '__len__': lambda self: len(self.buffer)
        })()

        # ✅ 添加情感模块（可选注入）
        self.emotion_module = EmotionModuleNone()

    def denormalize_action(self, norm_action):
        return norm_action * self.scales + self.offsets

    def normalize_action(self, real_action):
        return (real_action - self.offsets) / self.scales

    def act(self, state, deterministic=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state)
                action = torch.tanh(mean).cpu().numpy()[0]
            else:
                action, _ = self.policy.sample(state)
                action = action.cpu().numpy()[0]
        return self.denormalize_action(action)

    def remember(self, s, a, r, s_, d):
        norm_a = self.normalize_action(a)
        self.memory.buffer.append((s, norm_a, r, s_, d))

    def update(self):
        if len(self.memory.buffer) < self.batch_size:
            return

        batch = random.sample(self.memory.buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # === 更新 Critic ===
        with torch.no_grad():
            next_action, log_pi_next = self.policy.sample(s_)
            q1_next = self.q1_target(s_, next_action)
            q2_next = self.q2_target(s_, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = r + (1 - d) * self.gamma * (min_q_next - self.alpha * log_pi_next)

        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = nn.MSELoss()(q1, target_q)
        q2_loss = nn.MSELoss()(q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # === 更新 Policy ===
        new_action, log_pi = self.policy.sample(s)
        q1_new = self.q1(s, new_action)
        policy_loss = (self.alpha * log_pi - q1_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # === 不再更新 alpha（已固定） ===

        # === 更新 Target Q网络 ===
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item()
        }

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict()
        }, path)

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()


import os
import shutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging

# 日志配置
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/sac_training.log", filemode="w",
                    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_sac(env, agent, episodes=1000, max_steps=500, log_prefix="simple_sac_exp", model_path=None,
              pretrain_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/sac_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # 日志配置
    logger = logging.getLogger(f"sac_logger_{log_prefix}")
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
        print("pretrain_path:", pretrain_path)
        agent.load(pretrain_path)
        print(f"✅ 成功加载预训练模型: {pretrain_path}")
        logger.info(f"成功加载预训练模型: {pretrain_path}")
    else:
        print("❌ 未加载预训练模型，从头开始训练")
        logger.info("未加载预训练模型，从头开始训练")

    for ep in tqdm(range(episodes), desc="Simple SAC Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()
            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # 获取当前 emotion（无模块时默认返回）
            if hasattr(agent, "emotion_module"):
                emotion = agent.emotion_module.get_emotion()
            else:
                emotion = [0.5, 0.5, 0.0]

            # TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            if update_info:
                writer.add_scalar("Loss/Actor", update_info.get("policy_loss", 0.0), global_step)
                writer.add_scalar("Loss/Critic", update_info.get("q1_loss", 0.0), global_step)
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | "
                    f"Loss: {{'q1_loss': {update_info.get('q1_loss', 0.0):.2f}, 'policy_loss': {update_info.get('policy_loss', 0.0):.2f}}} | "
                    f"E: [{emotion[0]:.2f} {emotion[1]:.2f} {emotion[2]:.2f}]"
                )
            else:
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | Loss: None | "
                    f"E: [{emotion[0]:.2f} {emotion[1]:.2f} {emotion[2]:.2f}]"
                )

            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        # ✅ 每回合记录最新 emotion（确保是终态）
        if hasattr(agent, "emotion_module"):
            emotion = agent.emotion_module.get_emotion()
        else:
            emotion = [0.5, 0.5, 0.0]

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        actor_loss = update_info.get("policy_loss", "-") if update_info else "-"
        critic_loss = update_info.get("q1_loss", "-") if update_info else "-"

        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
            f"Emotion: [Curi: {emotion[0]:.2f}, Cons: {emotion[1]:.2f}, Anx: {emotion[2]:.2f}] | "
            f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
        )

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] 模型权重已保存为 {ckpt_path}")

    agent.save(final_model_path)
    logger.info(f"✅ 最终模型保存为 {final_model_path}")
    writer.close()
    return rewards
