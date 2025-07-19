import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import logging
import shutil
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

class DummyMemory:
    """
    A dummy memory buffer that pre-populates a fixed number of experience tuples.
    Each experience consists of a fake state, fake action, zero reward, fake next state,
    and a done flag set to False.
    This is useful for compatibility with algorithms or functions expecting a replay buffer.
    """

    def __init__(self, state_dim=2, action_dim=10):
        """
        Initialize the dummy memory with a fixed buffer of 20 identical fake experiences.

        Args:
            state_dim (int): Dimension of the state vector. Defaults to 2.
            action_dim (int): Dimension of the action vector. Defaults to 10.
        """
        fake_state = np.zeros(state_dim, dtype=np.float32)
        fake_action = np.zeros(action_dim, dtype=np.float32)
        self.buffer = deque([
            (fake_state, fake_action, 0.0, fake_state, False)
            for _ in range(20)
        ])

    def __len__(self):
        """
        Return the number of experiences stored in the buffer.

        Returns:
            int: Current size of the buffer.
        """
        return len(self.buffer)


# ============================ Network ============================
class PPOActorCritic(nn.Module):
    """
    Proximal Policy Optimization (PPO) Actor-Critic Network.

    This network jointly models both the policy (actor) and the value function (critic).
    The actor outputs actions scaled by a Tanh activation, and the critic estimates
    the state value.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        hidden_size = 256  # Number of hidden units in each layer

        # Actor network: maps state to action, using Tanh to bound actions between [-1, 1]
        self.actor = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, action_dim),
            nn.Tanh()
        )

        # Critic network: maps state to scalar value estimate
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state):
        """
        Forward pass through the actor and critic networks.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            action (torch.Tensor): Output action tensor bounded by Tanh.
            value (torch.Tensor): Estimated state value (scalar).
        """
        action = self.actor(state)
        value = self.critic(state)
        return action, value


# ============================ PPO Agent ============================
class PPOAgent:
    """
    Proximal Policy Optimization (PPO) agent for continuous action environments.

    Args:
        env: The environment object with state_dim, action_dim, and bounds attributes.
        device (str): Device to run the model on ('cuda' or 'cpu').
        gamma (float): Discount factor for rewards.
        clip_eps (float): PPO clipping epsilon parameter.
        lam (float): Lambda parameter for Generalized Advantage Estimation (GAE).
    """

    def __init__(self, env, device='cuda', gamma=0.99, clip_eps=0.2, lam=0.95):
        self.env = env
        self.device = device
        self.gamma = gamma
        self.clip_eps = clip_eps
        self.lam = lam

        # Initialize PPO actor-critic network
        self.policy = PPOActorCritic(env.state_dim, env.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=3e-4)

        # Precompute offsets and scales for action normalization/denormalization
        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

        # Dummy memory buffer to maintain compatibility with experience replay interfaces
        self.memory = DummyMemory(state_dim=env.state_dim, action_dim=env.action_dim)

    def normalize_action(self, action):
        """
        Normalize an action from environment bounds to [-1, 1] scale.

        Args:
            action (np.array): Action in original scale.

        Returns:
            np.array: Normalized action.
        """
        return (action - self.offsets) / self.scales

    def denormalize_action(self, norm_action):
        """
        Convert a normalized action back to the original environment action scale.

        Args:
            norm_action (np.array): Action normalized to [-1, 1].

        Returns:
            np.array: Action clipped to environment bounds.
        """
        action = norm_action * self.scales + self.offsets
        lows = np.array([lo for lo, _ in self.env.bounds.values()])
        highs = np.array([hi for _, hi in self.env.bounds.values()])
        return np.clip(action, lows, highs)

    def select_action(self, state):
        """
        Select an action given the current state by forwarding through the policy network.

        Args:
            state (np.array): Current environment state.

        Returns:
            tuple: Real-scale action (np.array) and estimated state value (float).
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            norm_action, value = self.policy(state_tensor)
        real_action = self.denormalize_action(norm_action.cpu().numpy().squeeze())

        # Optional: Uncomment to check normalization error
        # recovered = self.normalize_action(real_action)
        # delta = recovered - norm_action.cpu().numpy().squeeze()
        # print("🚨 Normalization error:", np.round(delta, 4))

        return real_action, value.item()

    def compute_log_prob(self, state, norm_action):
        """
        Compute the log probability of a normalized action under the current policy.

        Args:
            state (np.array): State input.
            norm_action (np.array): Normalized action.

        Returns:
            float: Log probability of the action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)
        with torch.no_grad():
            mu, _ = self.policy(state_tensor)
            dist = torch.distributions.Normal(mu, 0.1)  # fixed std dev of 0.1
            log_prob = dist.log_prob(norm_action_tensor).sum(-1)
        return log_prob.item()

    def compute_gae(self, rewards, values, dones):
        """
        Compute Generalized Advantage Estimation (GAE) for advantage calculation.

        Args:
            rewards (list[float]): Rewards obtained at each timestep.
            values (list[float]): Estimated state values.
            dones (list[bool]): Done flags indicating episode ends.

        Returns:
            tuple: (advantages, returns) as torch.FloatTensors on the agent device.
        """
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
        """
        Update the PPO policy and value networks using collected batch data.

        Args:
            states (list[np.array]): Batch of states.
            actions (list[np.array]): Batch of environment actions (not normalized).
            old_log_probs (list[float]): Log probabilities of actions under old policy.
            returns (torch.FloatTensor): Discounted returns.
            advantages (torch.FloatTensor): Computed advantages.
            mini_batch_size (int): Mini-batch size for multiple SGD updates.
            epochs (int): Number of epochs to perform over the batch.
        """
        # Normalize actions for training
        norm_actions = np.array([self.normalize_action(a) for a in actions])

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(norm_actions).to(self.device)
        old_log_probs = torch.FloatTensor(np.array(old_log_probs)).unsqueeze(1).to(self.device)

        dataset_size = states.size(0)
        indices = np.arange(dataset_size)

        for _ in range(epochs):  # Multiple training epochs
            np.random.shuffle(indices)
            for start in range(0, dataset_size, mini_batch_size):
                end = start + mini_batch_size
                mb_idx = indices[start:end]

                mb_states = states[mb_idx]
                mb_actions = actions[mb_idx]
                mb_old_log_probs = old_log_probs[mb_idx]
                mb_returns = returns[mb_idx]
                mb_advantages = advantages[mb_idx]

                # Forward pass through policy
                action_preds, values = self.policy(mb_states)
                dist = torch.distributions.Normal(action_preds, 0.1)
                log_probs = dist.log_prob(mb_actions).sum(-1, keepdim=True)

                # Calculate PPO clipped surrogate objective
                ratio = torch.exp(log_probs - mb_old_log_probs)
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1 - self.clip_eps, 1 + self.clip_eps) * mb_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Critic loss as MSE between predicted values and returns
                critic_loss = nn.MSELoss()(values, mb_returns)

                loss = actor_loss + 0.5 * critic_loss

                # Optimize the policy and value networks
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

    def save(self, path):
        """
        Save the model state dict to a file.

        Args:
            path (str): File path to save the model.
        """
        torch.save(self.policy.state_dict(), path)

    def load(self, path):
        """
        Load the model state dict from a file.

        Args:
            path (str): File path to load the model from.
        """
        self.policy.load_state_dict(torch.load(path))
        self.policy.eval()


def train_ppo(env, agent, episodes=1000, max_steps=500, log_prefix="ppo_exp", model_path=None, logger=None):
    """
    Train a PPO agent in the given environment.

    Args:
        env: The environment to train in.
        agent: PPOAgent instance to train.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum steps per episode.
        log_prefix (str): Prefix for logging directories and files.
        model_path (str or None): Optional path to save intermediate model checkpoints.

    Returns:
        list: List of episode rewards collected during training.
    """
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # === Setup logging and model saving directories ===
    tb_log_dir = f"runs/{log_prefix}"
    saved_model_dir = f"saved_models/{log_prefix}"
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    # Remove existing tensorboard logs for a fresh start
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Setup logger for training info
    log_file = f"logs/{log_prefix}.log"
    logger = logging.getLogger(f"ppo_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    # logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(fh)

    # Define default model checkpoint path if not provided
    if model_path is None:
        model_path = os.path.join(saved_model_dir, f"{log_prefix}.pth")
    final_model_path = os.path.join(saved_model_dir, f"{log_prefix}_final.pth")

    # Main training loop over episodes
    for ep in tqdm(range(episodes), desc="PPO Training", ncols=100):
        state = env.reset()
        buffer = {"states": [], "actions": [], "rewards": [], "dones": [], "values": [], "log_probs": []}
        ep_reward = 0
        average_step = 0

        # Run one episode
        for step in range(max_steps):
            # Select action and value estimate from the agent
            action, value = agent.select_action(state)
            norm_action = agent.normalize_action(action)

            # Step the environment
            next_state, reward, done, info = env.step(action)

            # Store experience in buffer for training
            buffer["states"].append(state)
            buffer["actions"].append(action)
            buffer["rewards"].append(reward)
            buffer["dones"].append(done)
            buffer["values"].append(value)
            buffer["log_probs"].append(agent.compute_log_prob(state, norm_action))

            state = next_state
            ep_reward += reward

            # Log step-wise metrics to TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info["weight_error"], global_step)
            writer.add_scalar("Metric/Time", info["total_time"], global_step)
            writer.add_scalar("Metric/SlowWeight", info["action"].get("slow_weight", 0.0), global_step)

            global_step += 1
            average_step += 1

            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        # Compute advantages and returns for PPO update
        advantages, returns = agent.compute_gae(buffer["rewards"], buffer["values"], buffer["dones"])

        # Update agent with collected batch data
        agent.update(
            buffer["states"], buffer["actions"],
            buffer["log_probs"], returns, advantages,
            mini_batch_size=32, epochs=5
        )

        # Log episode-level metrics to TensorBoard
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        # Log episode summary to file
        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info['weight_error']:.2f}g | Time: {info['total_time']:.2f}s | "
            f"SlowWeight: {info['action'].get('slow_weight', 0.0):.2f}g"
        )

        # Save intermediate model every 20 episodes
        if ep % 20 == 0:
            torch.save(agent.policy.state_dict(), model_path)
            logger.info(f"[Episode {ep}] Intermediate model saved to {model_path}")

    # Save final trained model
    torch.save(agent.policy.state_dict(), final_model_path)
    logger.info(f"✅ Training complete, final model saved to {final_model_path}")

    writer.close()
    return rewards
