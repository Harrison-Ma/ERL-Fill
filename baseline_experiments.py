from DifferentModules.Fuzzy_PID_agent import FuzzyControllerAgent
from DifferentModules.sac_agent import SACAgent
from DifferentModules.ddpg_agent import DDPGAgent
from EmotionModule import EmotionModuleNone
from DifferentModules.td3_agent import TD3Agent
from DifferentModules.ERL_Fill_agent import EmotionSACAgent
from DifferentModules.td3_bc_agent import TD3BCAgent
from DifferentModules.ppo_agent import PPOAgent
from DifferentModules.cql_agent import CQLAgent
from DifferentModules.PPOL import PPOLagrangian
from DifferentModules.rls_pidagent import RLS_PIDAgent
from DifferentModules.ddpg_agent import train_ddpg
from DifferentModules.td3_agent import train_td3
from DifferentModules.td3_bc_agent import train_td3_bc
from DifferentModules.ERL_Fill_agent import train_erl_fill
from DifferentModules.ppo_agent import train_ppo
from DifferentModules.sac_agent import train_sac
from DifferentModules.cql_agent import train_cql
from DifferentModules.PPOL import train_ppo_lagrangian
from DifferentModules.rls_pidagent import train_rls_pid

# =========================
# Part 1: Testing Utilities
# =========================


def test_agent(agent, env, model_path, max_steps=100, episodes=10):
    """
    Test a trained agent in the environment.

    Args:
        agent: The trained agent instance.
        env: The environment instance.
        model_path (str): Path to the saved model weights.
        max_steps (int): Maximum steps per test episode.
        episodes (int): Number of test episodes to run.

    Returns:
        float: Average total reward over all test episodes.
    """
    # Load the trained model weights
    agent.load(model_path)
    total_reward = 0

    # Run multiple test episodes
    for ep in range(episodes):
        state = env.reset()
        ep_reward = 0

        # Run one episode until done or max_steps reached
        for step in range(max_steps):
            # Select action without exploration noise
            action = agent.act(state, add_noise=False)
            state, reward, done, _ = env.step(action)
            ep_reward += reward
            if done:
                break

        # Print episode reward
        print(f"🎯 Test Episode {ep + 1}: Reward = {ep_reward:.2f}")
        total_reward += ep_reward

    # Calculate and print average reward over all episodes
    avg_reward = total_reward / episodes
    print(f"✅ Avg Test Reward: {avg_reward:.2f}")
    return avg_reward


# =========================
# Part 2: Training Utilities
# =========================


def train_agent(env, agent, train_fn, episodes=1000, max_steps=100, model_path="model.pth", log_prefix="exp"):
    """
    General training function to train an agent using a specified training function.

    Args:
        env: The environment instance.
        agent: The agent instance to be trained.
        train_fn: The training function to use (e.g., train_td3 or train_td3_bc).
        episodes (int): Number of episodes to train.
        max_steps (int): Maximum steps per episode.
        model_path (str): Path to save the trained model weights.
        log_prefix (str): Prefix for logging and saving files.

    Returns:
        list: A list of total rewards obtained in each training episode.
    """
    print(f"\n🚀 Start training [{log_prefix}]...")
    # Run the training function and collect rewards
    rewards = train_fn(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)
    # Save the trained model weights
    agent.save(model_path)
    print(f"✅ Training complete. Model saved at {model_path}.")
    return rewards


# =========================
# Part 3: Algorithm Wrappers
# =========================


def build_ddpg_agent(env, device):
    """
    Build and return a DDPG agent with an emotion module attached.

    Args:
        env: The environment instance.
        device: The device to run the agent on ('cpu' or 'cuda').

    Returns:
        agent: An instance of DDPGAgent with EmotionModuleNone attached.
    """
    agent = DDPGAgent(env, device=device, use_td_error=False)
    agent.emotion = EmotionModuleNone()
    return agent


def build_td3_agent(env, device):
    """
    Build and return a TD3 agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of TD3Agent.
    """

    return TD3Agent(env, device=device)


def build_td3_bc_agent(env, device):
    """
    Build and return a TD3+BC (Behavior Cloning) agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of TD3BCAgent.
    """
    return TD3BCAgent(env, device=device)


def build_ppo_agent(env, device):
    """
    Build and return a PPO agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of PPOAgent.
    """
    return PPOAgent(env, device=device)


def build_sac_agent(env, device):
    """
    Build and return a SAC agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of SACAgent.
    """
    return SACAgent(env, device=device)


def build_cql_agent(env, device):
    """
    Build and return a CQL (Conservative Q-Learning) agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of CQLAgent.
    """
    return CQLAgent(env, device=device)


def build_ppol_agent(env, device):
    """
    Build and return a PPOLagrangian agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of PPOLagrangian.
    """
    return PPOLagrangian(env, device=device)


def build_rls_pid_agent(env, device):
    """
    Build and return an RLS_PID agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on.

    Returns:
        agent: An instance of RLS_PIDAgent.
    """
    return RLS_PIDAgent(env, device=device)


def build_erl_fill_agent(env, device, lambda_emo=0.05):
    """
    Create and return an Emotion-SAC agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on ('cpu' or 'cuda').
        lambda_emo: Coefficient for the emotion component in the agent's loss (default: 0.05).

    Returns:
        agent: An instance of EmotionSACAgent with specified emotion coefficient.
    """
    agent = EmotionSACAgent(env, lambda_emo=lambda_emo, device=device)
    # agent.emotion = EmotionModule(device=device)  # Manually attach emotion module if needed
    return agent


def build_fuzzy_agent(env, device='cpu'):
    """
    Build and return a fuzzy logic controller agent.

    Args:
        env: The environment instance.
        device: The device to run the agent on (default is 'cpu').

    Returns:
        agent: An instance of FuzzyControllerAgent.
    """
    return FuzzyControllerAgent(env, device)


# === Trainer functions ===
def train_ddpg(env, agent, episodes=1000, max_steps=100, log_prefix="ddpg"):
    """
    Train a DDPG agent.

    Args:
        env: The environment instance.
        agent: The DDPG agent to train.
        episodes: Number of training episodes.
        max_steps: Maximum steps per episode.
        log_prefix: Prefix for logging files.

    Returns:
        Training rewards history.
    """
    return train_ddpg(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)


def train_td3(env, agent, episodes=1000, max_steps=100, log_prefix="td3"):
    """
    Train a TD3 agent.

    Args and returns same as train_ddpg.
    """
    return train_td3(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)


def train_td3_bc(env, agent, episodes=1000, max_steps=100, log_prefix="td3_bc"):
    """
    Train a TD3+BC (Behavior Cloning) agent.

    Args and returns same as train_ddpg.
    """
    return train_td3_bc(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix)


def train_erl_fill(env, agent, episodes=1000, max_steps=100, log_prefix="emotion_sac", lambda_emo=0.05):
    """
    Train an Emotion-SAC agent.

    Args:
        env: The environment instance.
        agent: The Emotion-SAC agent.
        episodes: Number of training episodes.
        max_steps: Max steps per episode.
        log_prefix: Prefix for logs.
        lambda_emo: Emotion loss coefficient.

    Returns:
        Training rewards history.
    """
    return train_erl_fill(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, lambda_emo=lambda_emo)


def train_ppo(env, agent, episodes=1000, max_steps=100, log_prefix="ppo", logger=None):
    """
    Train a PPO agent.

    Args and returns same as train_ddpg.
    """
    return train_ppo(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, logger=None)


def train_sac(env, agent, episodes=1000, max_steps=100, log_prefix="sac", pretrain_path=None, logger=None):
    """
    Train a SAC agent, optionally with pretrained weights.

    Args:
        pretrain_path: Path to pretrained model weights (optional).

    Other args and returns same as train_ddpg.
    """
    return train_sac(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, pretrain_path=pretrain_path, logger=None)


def train_cql(env, agent, episodes=1000, max_steps=100, log_prefix="cql", pretrain_path=None, logger=None):
    """
    Train a CQL agent, optionally with pretrained weights.

    Args and returns same as train_sac.
    """
    return train_cql(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, pretrain_path=pretrain_path, logger=logger)


def train_ppol(env, agent, episodes=1000, max_steps=100, log_prefix="ppol", pretrain_path=None):
    """
    Train a PPO Lagrangian agent, optionally with pretrained weights.

    Args and returns same as train_sac.
    """
    return train_ppo_lagrangian(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, pretrain_path=pretrain_path)


def train_rls_pid(env, agent, episodes=1000, max_steps=100, log_prefix="rls_pid", pretrain_path=None,logger=None):
    """
    Train an RLS_PID agent, optionally with pretrained weights.

    Args and returns same as train_sac.
    """
    return train_rls_pid(env, agent, episodes=episodes, max_steps=max_steps, log_prefix=log_prefix, pretrain_path=pretrain_path,logger=logger)


# =========================
# Part 4: Registries
# =========================

# === Agent builder registry ===
algo_builders = {
    # "er_ddpg": build_er_ddpg_agent,
    "ddpg": build_ddpg_agent,
    "td3": build_td3_agent,
    "td3_bc": build_td3_bc_agent,
    # "emotion_td3": build_emotion_td3_agent,  # ✅ Newly added
    "emotion_sac": build_erl_fill_agent,   # ← Added Emotion-SAC agent
    "ppo": build_ppo_agent,
    "sac": build_sac_agent,
    "cql": build_cql_agent,
    "ppol": build_ppol_agent,
    "rls_pid": build_rls_pid_agent
}

# === Trainer function registry ===
algo_trainers = {
    # "er_ddpg": train_ddpg,
    "ddpg": train_ddpg,
    "td3": train_td3,
    "td3_bc": train_td3_bc,
    # "emotion_td3": train_emotion_td3,  # ✅ Newly added
    "emotion_sac": train_erl_fill,
    "ppo": train_ppo,
    "sac": train_sac,
    "cql": train_cql,
    "ppol": train_ppol,
    "rls_pid": train_rls_pid
}

# === Function to get the corresponding agent builder by algorithm name ===


def get_agent_builder(algo_name):
    """
    Retrieve the agent builder function for a given algorithm name.

    Args:
        algo_name (str): Name of the RL algorithm.

    Returns:
        function: The builder function to create the agent.

    Raises:
        ValueError: If the algorithm name is not supported.
    """
    if algo_name not in algo_builders:
        raise ValueError(f"[get_agent_builder] Unsupported algorithm: {algo_name}")
    return algo_builders[algo_name]
