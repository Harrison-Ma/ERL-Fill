import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging
import shutil

# ============================ Logging Setup ============================
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/ppo_training.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class DummyMemory:
    def __init__(self, state_dim=2, action_dim=10):
        fake_state = np.zeros(state_dim, dtype=np.float32)
        fake_action = np.zeros(action_dim, dtype=np.float32)
        self.buffer = deque([
            (fake_state, fake_action, 0.0, fake_state, False)
            for _ in range(20)
        ])

    def __len__(self):
        return len(self.buffer)


# ============================ Network ============================
class PPOActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_size = 256
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, action_dim), nn.Tanh()
        )
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, hidden_size), nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        action = self.actor(state)
        value = self.critic(state)
        return action, value


# ============================ PPO Agent ============================
class PPOAgent:
    def __init__(self, env, device='cuda', gamma=0.99, clip_eps=0.2, lam=0.95):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lam = lam
        self.policy = PPOActorCritic(env.state_dim, env.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

        self.memory = DummyMemory(state_dim=env.state_dim, action_dim=env.action_dim)  # ✅ 添加这一行

    def normalize_action(self, action):
        return (action - self.offsets) / self.scales

    def denormalize_action(self, norm_action):
        action = norm_action * self.scales + self.offsets
        lows = np.array([lo for lo, _ in self.env.bounds.values()])
        highs = np.array([hi for _, hi in self.env.bounds.values()])
        return np.clip(action, lows, highs)

    def select_action(self, state):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            norm_action, value = self.policy(state_tensor)
        # print("归一化前的action：",norm_action)
        real_action = self.denormalize_action(norm_action.cpu().numpy().squeeze())
        # print("归一化后的action：", real_action)
        # ✅ 在 select_action() 返回之后加：
        # real_action = self.denormalize_action(norm_action.cpu().numpy().squeeze())
        # recovered = self.normalize_action(real_action)
        #
        # # ⚠️ 用于检测误差
        # delta = recovered - norm_action.cpu().numpy().squeeze()
        # print("🚨 归一化误差:", np.round(delta, 4))
        return real_action, value.item()

    def compute_log_prob(self, state, norm_action):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.policy(state_tensor)
            dist = torch.distributions.Normal(mu, 0.1)
            log_prob = dist.log_prob(norm_action_tensor).sum(-1)
        return log_prob.item()

    def compute_gae(self, rewards, values, dones):
        advantages, returns = [], []
        gae = 0
        next_value = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae)
            next_value = values[i]
            returns.insert(0, gae + values[i])
        return (
            torch.FloatTensor(advantages).unsqueeze(1).to(self.device),
            torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        )

    def update(self, states, actions, old_log_probs, returns, advantages, mini_batch_size=64, epochs=10):
        norm_actions = np.array([self.normalize_action(a) for a in actions])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(norm_actions).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(epochs):  # 多轮更新
            np.random.shuffle(indices)

            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                action_preds, values = self.policy(mb_states)
                dist = torch.distributions.Normal(action_preds, 0.1)
                log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)

                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                critic_loss = nn.MSELoss()(values, mb_returns)
                loss = actor_loss + 0.5 * critic_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, path):
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()


def train_ppo(env, agent, episodes=1000, max_steps=500, log_prefix="ppo_exp", model_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # === 路径设置 ===
    tb_log_dir = f"runs/ppo/{log_prefix}"
    saved_model_dir = f"saved_models/ppo/{log_prefix}"
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # === 日志设置 ===
    log_file = f"logs/ppo_training_{log_prefix}.log"
    logger = logging.getLogger(f"ppo_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # ✅ 模型保存路径
    if model_path is None:
        model_path = os.path.join(saved_model_dir, f"ppo_{log_prefix}_latest.pth")
    final_model_path = os.path.join(saved_model_dir, f"ppo_final_{log_prefix}.pth")

    for ep in tqdm(range(episodes), desc="PPO Training", ncols=100):
        state = env.reset()
        buffer = {"states": [], "actions": [], "rewards": [], "dones": [], "values": [], "log_probs": []}
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action, value = agent.select_action(state)
            norm_action = agent.normalize_action(action)
            next_state, reward, done, info = env.step(action)

            buffer["states"].append(state)
            buffer["actions"].append(action)
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)
            buffer["values"].append(value)
            buffer["log_probs"].append(agent.compute_log_prob(state, norm_action))

            state = next_state
            ep_reward += reward

            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info["weight_error"], global_step)
            writer.add_scalar("Metric/Time", info["total_time"], global_step)
            writer.add_scalar("Metric/SlowWeight", info["action"].get("slow_weight", 0.0), global_step)

            global_step += 1
            average_step += 1
            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        advantages, returns = agent.compute_gae(buffer["rewards"], buffer["values"], buffer["dones"])
        agent.update(
            buffer["states"], buffer["actions"],
            buffer["log_probs"], returns, advantages,
            mini_batch_size=32, epochs=5
        )

        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info['weight_error']:.2f}g | Time: {info['total_time']:.2f}s | "
            f"SlowWeight: {info['action'].get('slow_weight', 0.0):.2f}g"
        )

        # ✅ 每 20 轮保存一次中间模型
        if ep % 20 == 0:
            torch.save(agent.policy.state_dict(), model_path)
            logger.info(f"[Episode {ep}] 中间模型已保存至 {model_path}")

    # ✅ 最终保存
    torch.save(agent.policy.state_dict(), final_model_path)
    logger.info(f"✅ 训练完成，最终模型已保存至 {final_model_path}")

    writer.close()
    return rewards
