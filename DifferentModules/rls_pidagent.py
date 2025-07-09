import numpy as np

class DummyBuffer:
    def __init__(self):
        self.buffer = []

    def append(self, item):
        self.buffer.append(item)

    def __len__(self):
        return len(self.buffer)

    def __getitem__(self, idx):
        return self.buffer[idx]


class RLS_PIDAgent:
    def __init__(self, env, device=None, lambda_=0.99, delta=1e5):
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        # ✅ RLS系统辨识（状态+动作→预测next_state[0]）
        self.rls_dim = self.state_dim + self.action_dim + 1
        self.theta_rls = np.zeros((self.rls_dim, 1))
        self.P_rls = np.eye(self.rls_dim) * delta
        self.lambda_rls = lambda_

        # ✅ PID控制参数
        self.Kp = 1.0
        self.Ki = 0.1
        self.Kd = 0.05

        self.integral = 0.0
        self.prev_error = 0.0
        self.memory = DummyBuffer()

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

    def reset(self):
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, state, deterministic=True):
        # === PID 计算 ===
        target = getattr(self.env, 'target_weight', 25000)
        current_weight = state[0]
        error = target - current_weight

        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        # === 生成动作 ===
        action = np.clip(np.ones(self.action_dim) * output, -1, 1)
        action = self.denormalize_action(action)

        # === 安全性筛选 ===
        predicted_next = self.predict_next_state(state, action)
        safe_threshold = getattr(self.env, 'error_tolerance', 50)
        if abs(target - predicted_next) > 2 * safe_threshold:
            action = self.safe_action(state)

        return action

    def predict_next_state(self, state, action):
        x_rls = np.hstack([state, action, 1.0]).reshape(-1, 1)
        pred = float(x_rls.T @ self.theta_rls)
        return pred

    def safe_action(self, state):
        return np.zeros(self.action_dim)

    def remember(self, state, action, reward, next_state, done, cost):
        x_rls = np.hstack([state, action, 1.0]).reshape(-1, 1)
        y_rls = np.array([next_state[0]]).reshape(1, 1)

        Px = self.P_rls @ x_rls
        gain = Px / (self.lambda_rls + x_rls.T @ Px)
        self.theta_rls += gain @ (y_rls - x_rls.T @ self.theta_rls)
        self.P_rls = (self.P_rls - gain @ x_rls.T @ self.P_rls) / self.lambda_rls

        self.memory.append((state, action, reward, next_state, done, cost))

    def update(self):
        return {
            'rls_theta_norm': np.linalg.norm(self.theta_rls),
            'integral_term': self.integral,
            'last_error': self.prev_error
        }

    def save(self, path):
        np.savez(path,
                 theta_rls=self.theta_rls,
                 P_rls=self.P_rls,
                 Kp=self.Kp,
                 Ki=self.Ki,
                 Kd=self.Kd)

    def load(self, path):
        data = np.load(path)
        self.theta_rls = data['theta_rls']
        self.P_rls = data['P_rls']
        self.Kp = float(data['Kp'])
        self.Ki = float(data['Ki'])
        self.Kd = float(data['Kd'])


import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter

def train_rls_pid(env, agent, episodes=1000, max_steps=500, log_prefix="rls_pid_exp", model_path=None, pretrain_path=None):
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/rls_pid_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.npz")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.npz")

    logger = logging.getLogger(f"rls_pid_logger_{log_prefix}")
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

    for ep in tqdm(range(episodes), desc="RLS+PID Training", ncols=100):
        state = env.reset()
        agent.reset()
        ep_reward = 0
        average_step = 0
        unsafe_action_count = 0

        for step in range(max_steps):
            action = agent.act(state)
            predicted_next = agent.predict_next_state(state, action)
            target = getattr(env, 'target_weight', 25000)
            safe_threshold = getattr(env, 'error_tolerance', 50)
            if abs(target - predicted_next) > 2 * safe_threshold:
                unsafe_action_count += 1

            next_state, reward, done, info = env.step(action)
            cost = info.get("constraint_cost", 0.0)
            agent.remember(state, action, reward, next_state, done, cost)

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Cost/Step", cost, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)

            if done:
                break

        update_info = agent.update()
        avg_ep_reward = ep_reward / average_step if average_step > 0 else 0
        rewards.append(ep_reward)

        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", avg_ep_reward, ep)
        writer.add_scalar("Safety/UnsafeActions", unsafe_action_count, ep)

        if update_info:
            writer.add_scalar("Theta/RLS_Norm", update_info.get('rls_theta_norm', 0.0), ep)
            writer.add_scalar("PID/Integral", update_info.get('integral_term', 0.0), ep)
            writer.add_scalar("PID/LastError", update_info.get('last_error', 0.0), ep)

        logger.info(
            f"[Ep {ep:04d}] Reward: {ep_reward:.2f} | Avg: {avg_ep_reward:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
            f"UnsafeActions: {unsafe_action_count}"
        )

        if update_info:
            logger.info(
                f"RLS θ-norm: {update_info.get('rls_theta_norm', 0):.4f} | "
                f"Integral: {update_info.get('integral_term', 0):.2f} | LastError: {update_info.get('last_error', 0):.2f}"
            )

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.npz")
            agent.save(ckpt_path)
            logger.info(f"[Ep {ep:04d}] 模型已保存: {ckpt_path}")

    agent.save(final_model_path)
    logger.info(f"✅ 最终模型保存为 {final_model_path}")
    writer.close()
    return rewards


