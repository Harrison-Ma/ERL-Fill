import torch
import numpy as np
from WeightEnv import WeightEnv

# =========================
# Part 1: Testing Utilities
# =========================

def test_agent(agent, env, model_path, max_steps=100, episodes=10):
    agent.load(model_path)
    total_reward = 0
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0
        for step in range(max_steps):
            action = agent.act(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break
        print(f"🎯 Test Episode {ep + 1}: Reward = {ep_reward:.2f}")
        total_reward += ep_reward
    avg_reward = total_reward / episodes
    print(f"✅ Avg Test Reward: {avg_reward:.2f}")
    return avg_reward

# =========================
# Part 2: Training Utilities
# =========================

def train_agent(env, agent, train_fn, episodes=1000, max_steps=100, model_path="model.pth", log_prefix="exp"):
    print(f"\n🚀 Start training [{log_prefix}]...")
    rewards = train_fn(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
    agent.save(model_path)
    print(f"✅ Training complete. Model saved at {model_path}.")
    return rewards

# =========================
# Part 3: Algorithm Wrappers
# =========================

# # === 构建器 ===
# def build_er_ddpg_agent(env, device):
#     from WeightDemo import DDPGAgent, EmotionModule
#     agent = DDPGAgent(env, device=device, use_td_error=True)
#     agent.emotion = EmotionModule(device=device)
#     return agent

def build_ddpg_agent(env, device):
    from DifferentModules.ddpg_agent import DDPGAgent
    from EmotionModule import EmotionModuleNone
    agent = DDPGAgent(env, device=device, use_td_error=False)
    agent.emotion = EmotionModuleNone()
    return agent

# def build_td3_agent(env, device):
#     from td3_module import TD3Agent
#     return TD3Agent(env, device=device)

def build_td3_bc_agent(env, device):
    from DifferentModules.td3_bc_agent import TD3BCAgent
    return TD3BCAgent(env, device=device)

# def build_emotion_td3_agent(env, device):
#     from Emotion_TD3 import EmotionTD3Agent
#     return EmotionTD3Agent(env, device=device)

# def build_ppo_agent(env, device):
#     from ppo_module import PPOAgent
#     return PPOAgent(env, device=device)
#
# def build_sac_agent(env, device):
#     from sac_module import SACAgent
#     return SACAgent(env, device=device)

def build_cql_agent(env, device):
    from DifferentModules.cql_agent import CQLAgent
    return CQLAgent(env, device=device)

def build_ppol_agent(env, device):
    from DifferentModules.PPOL import PPOLagrangian
    return PPOLagrangian(env, device=device)

def build_rls_pid_agent(env, device):
    from DifferentModules.rls_pidagent import RLS_PIDAgent
    return RLS_PIDAgent(env, device=device)

def build_emotion_sac_agent(env, device,lambda_emo=0.05):
    """
    创建 Emotion-SAC Agent
    """
    from DifferentModules.ERL_Fill_agent import EmotionSACAgent  # 记得用你的 emotion 版SAC模块
    # from WeightDemo import EmotionModule  # 依赖情感模块

    agent = EmotionSACAgent(env,lambda_emo=lambda_emo, device=device)
    # agent.emotion = EmotionModule(device=device)  # 手动挂载情感模块
    return agent

# from fuzzy import FuzzyControllerAgent
#
# def build_fuzzy_agent(env, device='cpu'):
#     return FuzzyControllerAgent(env, device)

# === 训练器 ===
def train_ddpg(env, agent, episodes=1000, max_steps=100, log_prefix="ddpg"):
    from DifferentModules.ddpg_agent import train_ddpg
    return train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

# def train_td3(env, agent, episodes=1000, max_steps=100, log_prefix="td3"):
#     from td3_module import train_td3
#     return train_td3(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

def train_td3_bc(env, agent, episodes=1000, max_steps=100, log_prefix="td3_bc"):
    from DifferentModules.td3_bc_agent import train_td3_bc
    return train_td3_bc(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

# def train_emotion_td3(env, agent, episodes=1000, max_steps=100, log_prefix="emotion_td3"):
#     from Emotion_TD3 import train_emotion_td3
#     return train_emotion_td3(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

def train_emotion_sac(env, agent, episodes=1000, max_steps=100, log_prefix="emotion_sac", lambda_emo=0.05):
    from DifferentModules.ERL_Fill_agent import train_emotion_sac
    return train_emotion_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, lambda_emo=lambda_emo)

# def train_simple_sac(env, agent, episodes=1000, max_steps=100, log_prefix="emotion_sac"):
#     from simple_emotion_sac import train_simple_sac
#     return train_simple_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
#
# def train_ppo(env, agent, episodes=1000, max_steps=100, log_prefix="ppo"):
#     from ppo_module import train_ppo
#     return train_ppo(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)

# def train_sac(env, agent, episodes=1000, max_steps=100, log_prefix="sac",pretrain_path = None):
#     from sac_module import train_sac
#     return train_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix,pretrain_path = pretrain_path)

def train_cql(env, agent, episodes=1000, max_steps=100, log_prefix="cql",pretrain_path = None):
    from DifferentModules.cql_agent import train_cql
    return train_cql(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix,pretrain_path = pretrain_path)

def train_ppol(env, agent, episodes=1000, max_steps=100, log_prefix="ppol",pretrain_path = None):
    from DifferentModules.PPOL import train_ppo_lagrangian
    return train_ppo_lagrangian(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix,pretrain_path = pretrain_path)

def train_rls_pid(env, agent, episodes=1000, max_steps=100, log_prefix="rls_pid",pretrain_path = None):
    from DifferentModules.rls_pidagent import train_rls_pid
    return train_rls_pid(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix,pretrain_path = pretrain_path)
# =========================
# Part 4: Registries
# =========================

# === 构造器注册表 ===
algo_builders = {
    # "er_ddpg": build_er_ddpg_agent,
    "ddpg": build_ddpg_agent,
    # "td3": build_td3_agent,
    "td3_bc": build_td3_bc_agent,
    # "emotion_td3": build_emotion_td3_agent,  # ✅ 新增
    "emotion_sac": build_emotion_sac_agent,   # ← 加上Emotion-SAC
    # "ppo": build_ppo_agent,
    # "sac": build_sac_agent,
    "cql": build_cql_agent,
    "ppol": build_ppol_agent,
    "rls_pid": build_rls_pid_agent
}

# === 训练器注册表 ===
algo_trainers = {
    # "er_ddpg": train_ddpg,
    "ddpg": train_ddpg,
    # "td3": train_td3,
    "td3_bc": train_td3_bc,
    # "emotion_td3": train_emotion_td3,  # ✅ 新增
    "emotion_sac": train_emotion_sac,
    # "ppo": train_ppo,
    # "sac": train_sac,
    "cql": train_cql,
    "ppol": train_ppol,
    "rls_pid": train_rls_pid
}

# === 单独导出构造器的函数 ===
def get_agent_builder(algo_name):
    if algo_name not in algo_builders:
        raise ValueError(f"[get_agent_builder] Unsupported algorithm: {algo_name}")
    return algo_builders[algo_name]
