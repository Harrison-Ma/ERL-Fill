import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import shutil
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# === Emotion Replay Buffer ===
class EmotionReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

class EmotionGaussianPolicy(nn.Module):
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # 基础输出
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

        # === 情绪调制模块（用于调节 mean / log_std）===
        self.emotion_adapter_mean = nn.Sequential(
            nn.Linear(emotion_dim, action_dim),
            nn.Tanh()  # 输出范围在 [-1,1]
        )

        self.emotion_adapter_std = nn.Sequential(
            nn.Linear(emotion_dim, action_dim),
            nn.Tanh()  # 可调节方差大小
        )

    def forward(self, state, emotion):
        state_feat = self.state_encoder(state)
        emo_feat = self.emotion_encoder(emotion)
        x = self.policy_net(torch.cat([state_feat, emo_feat], dim=-1))

        base_mean = self.mean_layer(x)
        base_log_std = self.log_std_layer(x)

        # === 情绪调制 ===
        mean_adjust = self.emotion_adapter_mean(emotion)
        std_adjust = self.emotion_adapter_std(emotion)

        mean = base_mean + mean_adjust
        log_std = base_log_std + std_adjust
        log_std = torch.clamp(log_std, -20, 2)  # 保持稳定

        return mean, log_std

    def sample(self, state, emotion):
        mean, log_std = self(state, emotion)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization
        action = torch.tanh(z)

        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def get_action_entropy(self, state, emotion):
        mean, log_std = self(state, emotion)
        std = torch.exp(log_std)
        entropy = log_std + 0.5 * np.log(2 * np.pi * np.e)  # per-dimension entropy
        return entropy.sum(dim=-1).mean().item()  # mean over batch

# === Emotion-Aware Q Network ===
class EmotionAwareQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256):
        super().__init__()
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        self.q_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action, emotion):
        s_feat = self.state_encoder(state)
        a_feat = self.action_encoder(action)
        e_feat = self.emotion_encoder(emotion)
        x = torch.cat([s_feat, a_feat, e_feat], dim=-1)
        return self.q_net(x)

# === EmotionSAC Agent ===
class EmotionSACAgent:
    def __init__(self, env, lambda_emo=0.05, device='cuda'):
        self.env = env
        self.device = device
        self.lambda_emo = lambda_emo


        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.gamma = 0.99
        self.tau = 0.005
        self.alpha = 0.3
        self.batch_size = 128

        self.train_step = 1

        from EmotionModule import EmotionModule, save_emotion_module, load_emotion_module
        self.emotion = EmotionModule(device=device)
        self.save_emotion_module = save_emotion_module
        self.load_emotion_module = load_emotion_module

        self.policy = EmotionGaussianPolicy(self.state_dim, self.action_dim).to(device)
        self.q1 = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q2 = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q1_target = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q2_target = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        self.memory = EmotionReplayBuffer()

        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

    def normalize_action(self, action):
        return (action - self.offsets) / self.scales

    def denormalize_action(self, norm_action):
        return norm_action * self.scales + self.offsets

    def act(self, state, deterministic=False, add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        emotion = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_tensor, emotion)
                action = torch.tanh(mean)
            elif add_noise:
                action, _ = self.policy.sample(state_tensor, emotion)
            else:
                mean, _ = self.policy(state_tensor, emotion)
                action = torch.tanh(mean)

        action = action.cpu().numpy().squeeze()
        return self.denormalize_action(action)

    def remember(self, s, a, r, s_, d):
        norm_a = self.normalize_action(a)
        self.memory.add((s, norm_a, r, s_, d, self.emotion.get_emotion()))

    def update(self):
        if len(self.memory) < self.batch_size:
            return None

        batch = self.memory.sample(self.batch_size)
        s, a, r, s_, d, e = zip(*batch)

        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        e = torch.FloatTensor(e).to(self.device)

        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(s_, e)
            target_q1 = self.q1_target(s_, next_action, e)
            target_q2 = self.q2_target(s_, next_action, e)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = r + (1 - d) * self.gamma * min_target_q

        current_q1 = self.q1(s, a, e)
        current_q2 = self.q2(s, a, e)
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)

        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        new_action, log_pi = self.policy.sample(s, e)
        q1_new = self.q1(s, new_action, e)
        policy_loss = (self.alpha * log_pi - q1_new).mean()

        # === �� 情绪正则项 ===
        # 记录前一轮情绪
        e_now = self.emotion.get_emotion()  # numpy, shape=(3,)
        if not hasattr(self, 'prev_emotion'):
            self.prev_emotion = e_now

        e_now_tensor = torch.FloatTensor(e_now).to(self.device)
        e_prev_tensor = torch.FloatTensor(self.prev_emotion).to(self.device)
        emo_reg = torch.norm(e_now_tensor - e_prev_tensor, p=2)

        # lambda_emo = 0.05  # �� 可调超参数
        total_policy_loss = policy_loss + self.lambda_emo * emo_reg
        print("self.lambda_emo:###############",self.lambda_emo)

        # 更新记录
        self.prev_emotion = e_now

        self.policy_optim.zero_grad()
        total_policy_loss.backward()
        self.policy_optim.step()

        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "emo_reg": emo_reg.item()  # 可以加到 TensorBoard 日志中观察
        }

    def save(self, path):
        torch.save({
            "policy": self.policy.state_dict()
        }, path)
        # self.save_emotion_module(self.emotion, path.replace(".pth", "_emo.pth"))
        if hasattr(self.emotion, 'transformer'):
            self.save_emotion_module(self.emotion, path.replace(".pth", "_emo.pth"))

    def load(self, path):
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()

        # === 判断是否有 transformer 才加载情感模块 ===
        if hasattr(self.emotion, 'transformer'):
            emo_path = path.replace(".pth", "_emo.pth")
            if os.path.exists(emo_path):
                self.load_emotion_module(self.emotion, emo_path)
                self.emotion.transformer.eval()
                print(f"[Info] Emotion transformer weights loaded from: {emo_path}")
            else:
                print(f"[Warning] Emotion weights not found at: {emo_path}. Skipped.")
        else:
            print(f"[Info] Current emotion module [{type(self.emotion).__name__}] has no transformer. Skip loading.")

from CommonInterface.Logger import init_logger
import time
import logging

# 日志配置
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename="logs/sac_training_transformer.log", filemode="w",
                    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def train_emotion_sac(env, agent, episodes=1000, max_steps=500, log_prefix="emotion_sac", model_path=None, logger=None,lambda_emo=0.05):
    if logger is None:
        logger = init_logger(log_prefix=log_prefix)

    agent.lambda_emo = lambda_emo
    print("agent.lambda_emo",agent.lambda_emo)
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    tb_log_dir = f"runs/{log_prefix}"
    saved_model_dir = f"saved_models/{log_prefix}"
    os.makedirs(saved_model_dir, exist_ok=True)

    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    if model_path is None:
        model_path = os.path.join(saved_model_dir, f"{log_prefix}_final.pth")

    start_100 = time.time()

    metrics_summary = []

    for ep in tqdm(range(episodes), desc="EmotionSAC-Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0
        ep_err = 0
        ep_time = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # ✅ 添加这行（确保 emotion 状态会变）
            agent.emotion.update(reward, action)

            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()

            state = next_state
            ep_reward += reward
            ep_err += info.get("weight_error", 0.0)
            ep_time += info.get("total_time", 0.0)
            average_step += 1

            # === Logging scalar values ===
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            emotion_state = agent.emotion.get_emotion()
            writer.add_scalar("Emotion/Curiosity", float(emotion_state[0]), global_step)
            writer.add_scalar("Emotion/Conservativeness", float(emotion_state[1]), global_step)
            writer.add_scalar("Emotion/Anxiety", float(emotion_state[2]), global_step)


            # if update_info:
            #     writer.add_scalar("Loss/Actor", update_info.get("actor_loss", 0.0), global_step)
            #     writer.add_scalar("Loss/Critic", update_info.get("critic_loss", 0.0), global_step)
            
            if update_info:
                writer.add_scalar("Loss/Actor", update_info.get("policy_loss", 0.0), global_step)
                writer.add_scalar("Loss/Critic", update_info.get("q1_loss", 0.0), global_step)

            if step % (max_steps // 10) == 0 or step == max_steps - 1:
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | Loss: {update_info} | "
                    f"E: {np.round(emotion_state, 2)}"
                )

            global_step += 1
            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)
        writer.add_scalar("Metric/Weight", info.get("final_weight", 0.0), ep)


        # writer.add_scalar("Metrics/AvgReward", ep_reward / average_step, ep)
        # writer.add_scalar("Metrics/AvgError", ep_err / average_step,  ep)
        # writer.add_scalar("Metrics/CompletionTime", ep_time / average_step, ep)
        # reward_variance = np.var(rewards[-100:] if len(rewards) >= 100 else rewards)
        # writer.add_scalar("Metrics/RewardVariance", reward_variance, ep) #最后算
        # writer.add_scalar("Metrics/Entropy", action_entropy, ep)
        # 情感波动：当前情感 vs 上一回合
        if ep > 0:
            emotion_fluctuation = np.linalg.norm(np.array(emotion_state) - np.array(prev_emotion_state))
        else:
            emotion_fluctuation = 0.0
        prev_emotion_state = emotion_state.copy()
        writer.add_scalar("Metrics/EmotionFluctuation", emotion_fluctuation, ep)



        if update_info:
            writer.add_scalar("Loss/EmotionReg", update_info.get("emo_reg", 0.0), global_step)

        for key, val in info.get("action", {}).items():
            writer.add_scalar(f"Action/{key}", val, ep)

        actor_loss = update_info.get("policy_loss", "-") if update_info else "-"
        critic_loss = update_info.get("q1_loss", "-") if update_info else "-"
        logger.info(
            f"[Episode {ep:04d}] "
            f"TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
            f"Emotion: [Curi: {emotion_state[0]:.2f}, Cons: {emotion_state[1]:.2f}, Anx: {emotion_state[2]:.2f}] | "
            f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
        )

        if ep > 0 and ep % 100 == 0:
            duration_100 = time.time() - start_100
            logger.info(f"⏱️ 第 {ep-100}~{ep} 回合共耗时 {duration_100:.2f} 秒，平均 {duration_100 / 100:.2f} 秒/回合")
            start_100 = time.time()

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(saved_model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep}] 模型权重已保存为 {ckpt_path}")

        agent.train_step += 1

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            emotion_tensor = torch.FloatTensor(agent.emotion.get_emotion()).unsqueeze(0).to(agent.device)
            action_entropy = agent.policy.get_action_entropy(state_tensor, emotion_tensor)

        #量化指标评估
        # 指标1：整体平均奖励（收敛性）
        # 指标2：整体平均误差 （控制目标精度1）
        # 指标3：整体平均计量时间 （控制目标精度2）
        # 指标4：整体平均动作熵（搜索性）
        # 指标5：最快收敛代数 （效率）
        # 指标6：整体奖励方差（稳定性）
        # 指标7：情感波动标准差（动态寻优效果）
        writer.add_scalar("Metrics/AvgReward", ep_reward / average_step, ep)
        writer.add_scalar("Metrics/AvgError", ep_err / average_step,  ep)
        writer.add_scalar("Metrics/CompletionTime", ep_time / average_step, ep)
        reward_variance = np.var(rewards[-100:] if len(rewards) >= 100 else rewards)
        writer.add_scalar("Metrics/RewardVariance", reward_variance, ep) #最后算
        writer.add_scalar("Metrics/Entropy", action_entropy, ep)

        metrics_summary.append({
            "Episode": ep,
            "AvgReward": ep_reward / average_step,
            "Entropy": action_entropy,
            "FinalError_g": ep_err / average_step,
            "CompletionTime_s": ep_time / average_step,
            "RewardVariance": reward_variance,
            "EmotionFluctuation": emotion_fluctuation
        })
    agent.save(model_path)
    print(f"✅ Final model saved at {model_path}")
    logger.info(f"✅ 最终模型保存为 {model_path}")
    writer.close()
    import pandas as pd
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(f"{saved_model_dir}/{log_prefix}_metrics_summary_{lambda_emo}.csv", index=False)
    return rewards
