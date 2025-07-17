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
    """
    Replay buffer to store environment transitions for experience replay.

    Attributes:
        buffer (collections.deque): Double-ended queue with fixed maximum size to hold transitions.
    """

    def __init__(self, max_size=100000):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of transitions to store in the buffer.
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, *args):
        """
        Add a transition to the replay buffer.

        Args:
            *args: A tuple representing a transition, typically
                   (state, action, reward, next_state, done).
        """
        self.buffer.append(args)

    def __len__(self):
        """
        Get the current number of transitions stored in the buffer.

        Returns:
            int: Number of stored transitions.
        """
        return len(self.buffer)


class TD3Actor(nn.Module):
    """
    Actor network for TD3 algorithm, outputs continuous actions given states.

    Architecture:
        Fully connected neural network with two hidden layers (256 units each),
        ReLU activations, and Tanh activation at output to bound actions.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the output action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh()  # Action output bounded in [-1, 1]
        )

    def forward(self, state):
        """
        Forward pass through the actor network.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            torch.Tensor: Output action tensor of shape (batch_size, action_dim),
                          with values in range [-1, 1].
        """
        return self.net(state)


class TD3Critic(nn.Module):
    """
    Twin Critic network for TD3 algorithm, estimates Q-values for state-action pairs.

    Architecture:
        Two separate Q-networks (Q1 and Q2), each a fully connected neural network with
        two hidden layers (256 units each) and ReLU activations.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the input action.
    """

    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        self.q2 = nn.Sequential(
            nn.Linear(state_dim + action_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, state, action):
        """
        Forward pass through both critic networks to estimate Q-values.

        Args:
            state (torch.Tensor): Input state tensor of shape (batch_size, state_dim).
            action (torch.Tensor): Input action tensor of shape (batch_size, action_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Q-value estimates from Q1 and Q2 networks,
                                               each of shape (batch_size, 1).
        """
        sa = torch.cat([state, action], dim=1)  # Concatenate state and action along feature dimension
        return self.q1(sa), self.q2(sa)


class TD3Agent:
    """
    Twin Delayed Deep Deterministic Policy Gradient (TD3) agent.

    This agent interacts with the environment, stores experiences, and updates
    actor and critic networks to learn an optimal policy for continuous control tasks.

    Args:
        env: Environment instance with attributes `state_dim`, `action_dim`, and `bounds`.
        device (str): Device to run the model on ('cuda' or 'cpu').
    """

    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        # Initialize actor and critic networks along with their target networks
        self.actor = TD3Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target = TD3Actor(self.state_dim, self.action_dim).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        self.critic = TD3Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target = TD3Critic(self.state_dim, self.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers for actor and critic
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=1e-3)

        # Replay buffer to store experience tuples
        self.memory = ReplayBuffer()
        self.batch_size = 128
        self.gamma = 0.99           # Discount factor
        self.tau = 0.005            # Target network update rate
        self.policy_noise = 0.2     # Noise added to target policy during critic update
        self.noise_clip = 0.5       # Noise clipping range
        self.policy_delay = 2       # Frequency of delayed policy updates
        self.total_it = 0           # Total training iterations counter

    def normalize_action(self, action):
        """
        Normalize action from environment-specific bounds to [-1, 1].

        Args:
            action (array-like): Raw action in original bounds.

        Returns:
            np.ndarray: Normalized action within [-1, 1].
        """
        return np.array([
            2 * (action[i] - low) / (high - low) - 1
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def denormalize_action(self, norm_action):
        """
        Convert normalized action in [-1, 1] back to environment-specific bounds.

        Args:
            norm_action (array-like): Action normalized to [-1, 1].

        Returns:
            np.ndarray: Denormalized action within original bounds.
        """
        return np.array([
            0.5 * (norm_action[i] + 1) * (high - low) + low
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience tuple in replay buffer, normalizing action beforehand.

        Args:
            state (array-like): Current state.
            action (array-like): Action taken in the environment.
            reward (float): Reward received.
            next_state (array-like): Next state after taking action.
            done (bool or float): Whether episode ended after this step.
        """
        norm_action = self.normalize_action(action)
        self.memory.append(state, norm_action, reward, next_state, done)

    def select_action(self, state, noise_scale=0.1):
        """
        Select action for a given state, adding optional exploration noise.

        Args:
            state (array-like): Current state.
            noise_scale (float): Scale of Gaussian noise added to action for exploration.

        Returns:
            np.ndarray: Action in environment's original bounds.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action = self.actor(state_tensor).cpu().data.numpy().flatten()
        norm_action += noise_scale * np.random.randn(self.action_dim)
        norm_action = np.clip(norm_action, -1, 1)
        return self.denormalize_action(norm_action)

    def train_step(self):
        """
        Perform a single training step updating critic and actor networks.

        - Samples a batch from replay buffer.
        - Updates critic networks using TD3 clipped double-Q learning.
        - Updates actor network and target networks on a delayed schedule.

        Returns:
            Tuple:
                actor_loss (float or None): Loss value for actor network update or None if skipped.
                critic_loss (float): Loss value for critic networks update.
                q_value (float): Mean Q-value estimate from critic.
        """
        if len(self.memory) < self.batch_size:
            return None, None, None  # Not enough samples for training

        self.total_it += 1

        batch = random.sample(self.memory.buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)

        # Compute target Q-values with clipped noise for policy smoothing
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        # Compute current Q estimates and critic loss
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        # Update critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        q_value = q1.mean().item()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            actor_loss = -self.critic.q1(torch.cat([states, self.actor(states)], dim=1)).mean()
            self.actor_optim.zero_grad()
            actor_loss.backward()
            self.actor_optim.step()

            # Soft update target networks
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
        """
        Save actor and critic network weights to a file.

        Args:
            path (str): File path to save the model weights.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_weights(self, weights):
        """
        Load actor and critic weights from a state dictionary.

        Args:
            weights (dict): Dictionary containing 'actor' and 'critic' state_dict.
        """
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def load(self, path):
        """
        Load model weights from a file.

        Args:
            path (str): File path to load the model weights from.
        """
        weights = torch.load(path)
        self.load_weights(weights)

    def get_weights(self):
        """
        Retrieve current actor and critic network weights.

        Returns:
            dict: Dictionary containing 'actor' and 'critic' state_dict.
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def act(self, state, add_noise=False):
        """
        Return action for given state with optional exploration noise.

        Args:
            state (array-like): Current state.
            add_noise (bool): Whether to add exploration noise to action.

        Returns:
            np.ndarray: Action in original bounds.
        """
        noise = 0.1 if add_noise else 0.0
        return self.select_action(state, noise_scale=noise)


# Logging configuration
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(filename='logs/td3_training.log', filemode='w',
                    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def train_td3(env, agent, episodes=1000, max_steps=500, log_prefix="td3_exp", pretrain_path=None):
    """
    Train the TD3 agent on the given environment.

    Args:
        env: Environment instance with required methods and attributes.
        agent: TD3Agent instance with actor and critic networks.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum steps per episode.
        log_prefix (str): Prefix for logging and saving directories.
        pretrain_path (str or None): Path to pre-trained model checkpoint.

    Returns:
        list: Episode total rewards collected during training.
    """
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # === Set up directories for TensorBoard logs and saved models ===
    tb_log_dir = f"runs/td3/{log_prefix}"
    saved_model_dir = f"saved_models/td3/{log_prefix}"
    os.makedirs(tb_log_dir, exist_ok=True)
    os.makedirs(saved_model_dir, exist_ok=True)

    # Remove TensorBoard log directory if it already exists to avoid clutter
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # === Setup dedicated logger for this training run ===
    log_file = f"logs/td3_training_{log_prefix}.log"
    logger_td3 = logging.getLogger(f"td3_logger_{log_prefix}")
    logger_td3.setLevel(logging.INFO)
    logger_td3.propagate = False
    if not logger_td3.handlers:
        fh = logging.FileHandler(log_file, mode='w', encoding='utf-8')
        fh.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger_td3.addHandler(fh)

    # === Load pre-trained weights if path provided and exists ===
    use_pretrain = pretrain_path is not None and os.path.exists(pretrain_path)
    if use_pretrain:
        ckpt = torch.load(pretrain_path)
        agent.actor.load_state_dict(ckpt['actor'])
        agent.critic.load_state_dict(ckpt['critic'])
        print(f"✅ Successfully loaded pre-trained weights: {pretrain_path}")
        logger_td3.info(f"Successfully loaded pre-trained weights: {pretrain_path}")
    else:
        print("🚫 No pre-trained weights loaded, training from scratch.")
        logger_td3.info("No pre-trained weights loaded, training from scratch.")

    # === Main training loop over episodes ===
    for ep in tqdm(range(episodes), desc="TD3 Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            # Select action according to current policy with optional exploration noise
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action)

            # Store transition in replay buffer
            agent.remember(state, action, reward, next_state, float(done))

            # Perform one training step for actor and critic networks
            actor_loss, critic_loss, q_value = agent.train_step()

            state = next_state
            ep_reward += reward

            # Log training metrics to TensorBoard
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

        # Log episode summary statistics to TensorBoard
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        # Prepare loss strings for logging
        actor_loss_str = f"{actor_loss:.4f}" if actor_loss is not None else "-"
        critic_loss_str = f"{critic_loss:.4f}" if critic_loss is not None else "-"

        # Log detailed episode info to dedicated logger
        logger_td3.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info['weight_error']:.2f}g | Time: {info['total_time']:.2f}s | "
            f"SlowWeight: {info['action'].get('slow_weight', 0.0):.2f}g | "
            f"ActorLoss: {actor_loss_str} | CriticLoss: {critic_loss_str}"
        )

        # Save model checkpoints every 20 episodes and at the last episode
        if ep % 20 == 0 or ep == episodes - 1:
            model_path = os.path.join(saved_model_dir, f"td3_{log_prefix}_ep{ep:04d}.pth")
            torch.save({
                'actor': agent.actor.state_dict(),
                'critic': agent.critic.state_dict()
            }, model_path)
            logger_td3.info(f"[Episode {ep}] Model checkpoint saved at: {model_path}")

    # === Save final trained model ===
    final_path = os.path.join(saved_model_dir, f"td3_{log_prefix}_final.pth")
    torch.save({
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict()
    }, final_path)
    logger_td3.info(f"✅ Training complete, final model saved at: {final_path}")

    writer.close()
    return rewards

