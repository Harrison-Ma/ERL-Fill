import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import collections
import os
import shutil
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm


class ReplayBuffer:
    """
    Replay buffer to store and sample experience tuples for training.

    Attributes:
        buffer (collections.deque): Circular buffer storing experience tuples.
    """

    def __init__(self, max_size=100000):
        """
        Initialize the replay buffer.

        Args:
            max_size (int): Maximum number of experience tuples to store.
        """
        self.buffer = collections.deque(maxlen=max_size)

    def append(self, *args):
        """
        Append a new experience tuple to the buffer.

        Args:
            *args: Elements of the experience tuple (state, action, reward, next_state, done).
        """
        self.buffer.append(args)

    def __len__(self):
        """
        Return the current size of the buffer.

        Returns:
            int: Number of stored experience tuples.
        """
        return len(self.buffer)


class TD3Actor(nn.Module):
    """
    Actor network for TD3 algorithm that outputs continuous actions
    scaled between -1 and 1 using Tanh activation.

    Attributes:
        net (nn.Sequential): Neural network model for the actor.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the actor network.

        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the output action vector.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, action_dim), nn.Tanh()
        )

    def forward(self, state):
        """
        Forward pass to compute the action from the state.

        Args:
            state (torch.Tensor): Input state tensor.

        Returns:
            torch.Tensor: Output action tensor scaled between -1 and 1.
        """
        return self.net(state)


class TD3Critic(nn.Module):
    """
    Critic network for TD3 algorithm, consisting of two Q-functions
    for clipped double Q-learning.

    Attributes:
        q1 (nn.Sequential): First Q-function neural network.
        q2 (nn.Sequential): Second Q-function neural network.
    """

    def __init__(self, state_dim, action_dim):
        """
        Initialize the critic networks.

        Args:
            state_dim (int): Dimension of the input state vector.
            action_dim (int): Dimension of the input action vector.
        """
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
        """
        Forward pass to compute Q-values from state-action pairs.

        Args:
            state (torch.Tensor): Batch of states.
            action (torch.Tensor): Batch of actions.

        Returns:
            tuple: Two Q-value tensors, one from each Q-function.
        """
        sa = torch.cat([state, action], dim=1)  # Concatenate state and action along feature dimension
        return self.q1(sa), self.q2(sa)


class TD3BCAgent:
    """
    TD3 agent with Behavior Cloning (BC) regularization.
    This agent combines standard TD3 actor-critic training with
    an additional BC loss term to encourage the actor to imitate
    actions from the replay buffer.

    Attributes:
        env: Environment instance containing state/action specs.
        device (str): Device to run the model on ('cuda' or 'cpu').
        state_dim (int): Dimension of state space.
        action_dim (int): Dimension of action space.
        bounds (dict): Action bounds for normalization/denormalization.
        actor (TD3Actor): Actor network.
        actor_target (TD3Actor): Target actor network.
        critic (TD3Critic): Critic network with two Q-functions.
        critic_target (TD3Critic): Target critic network.
        actor_optim (torch.optim.Optimizer): Optimizer for actor.
        critic_optim (torch.optim.Optimizer): Optimizer for critic.
        memory (ReplayBuffer): Replay buffer for experience storage.
        batch_size (int): Mini-batch size for training.
        gamma (float): Discount factor.
        tau (float): Soft update coefficient.
        policy_noise (float): Noise added to target policy actions.
        noise_clip (float): Clipping range for policy noise.
        policy_delay (int): Delay for actor updates compared to critic.
        bc_coef (float): Coefficient for behavior cloning loss.
        total_it (int): Counter for total training iterations.
    """

    def __init__(self, env, device='cuda'):
        """
        Initialize TD3BCAgent with environment and device.

        Args:
            env: Environment with attributes state_dim, action_dim, bounds.
            device (str): Compute device.
        """
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
        self.bc_coef = 2.5  # Coefficient for behavior cloning loss
        self.total_it = 0

    def normalize_action(self, action):
        """
        Normalize action from environment bounds to [-1, 1].

        Args:
            action (array-like): Raw action values.

        Returns:
            np.ndarray: Normalized action values.
        """
        return np.array([
            2 * (action[i] - low) / (high - low) - 1
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def denormalize_action(self, norm_action):
        """
        Convert normalized action from [-1, 1] back to environment bounds.

        Args:
            norm_action (array-like): Normalized action values.

        Returns:
            np.ndarray: Denormalized action values.
        """
        return np.array([
            0.5 * (norm_action[i] + 1) * (high - low) + low
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def remember(self, state, action, reward, next_state, done):
        """
        Store experience tuple in replay buffer after normalizing action.

        Args:
            state: Current state.
            action: Raw action taken.
            reward: Reward received.
            next_state: Next state.
            done: Done flag (episode termination).
        """
        norm_action = self.normalize_action(action)
        self.memory.append(state, norm_action, reward, next_state, done)

    def select_action(self, state, noise_scale=0.1):
        """
        Select action for given state with optional exploration noise.

        Args:
            state: Current state.
            noise_scale (float): Std deviation of added Gaussian noise.

        Returns:
            np.ndarray: Denormalized action selected by actor.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        norm_action = self.actor(state_tensor).cpu().data.numpy().flatten()
        norm_action += noise_scale * np.random.randn(self.action_dim)
        norm_action = np.clip(norm_action, -1, 1)
        return self.denormalize_action(norm_action)

    def act(self, state, add_noise=False):
        """
        Convenient wrapper to select action with optional noise.

        Args:
            state: Current state.
            add_noise (bool): Whether to add exploration noise.

        Returns:
            np.ndarray: Action selected.
        """
        noise = 0.1 if add_noise else 0.0
        return self.select_action(state, noise_scale=noise)

    def train_step(self):
        """
        Perform a single training step updating critic and actor networks.

        Returns:
            tuple: (actor_loss, critic_loss, mean_q_value)
        """
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

        # Compute target Q value
        with torch.no_grad():
            noise = (torch.randn_like(actions) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            next_actions = (self.actor_target(next_states) + noise).clamp(-1, 1)
            q1_target, q2_target = self.critic_target(next_states, next_actions)
            q_target = rewards + self.gamma * (1 - dones) * torch.min(q1_target, q2_target)

        # Get current Q estimates
        q1, q2 = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q1, q_target) + nn.MSELoss()(q2, q_target)

        # Optimize critic
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        actor_loss = None
        q_value = q1.mean().item()

        # Delayed policy updates
        if self.total_it % self.policy_delay == 0:
            pred_action = self.actor(states)
            q_val = self.critic.q1(torch.cat([states, pred_action], dim=1))
            actor_loss_td3 = -q_val.mean()
            actor_loss_bc = nn.MSELoss()(pred_action, actions)
            actor_loss = actor_loss_td3 + self.bc_coef * actor_loss_bc

            # Optimize actor
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
        Save actor and critic model weights to a file.

        Args:
            path (str): File path to save weights.
        """
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }, path)

    def load_weights(self, weights):
        """
        Load actor and critic weights from a dictionary.

        Args:
            weights (dict): Dictionary containing actor and critic weights.
        """
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def load(self, path):
        """
        Load weights from a saved checkpoint file.

        Args:
            path (str): File path to checkpoint.
        """
        weights = torch.load(path)
        self.load_weights(weights)

    def get_weights(self):
        """
        Get current actor and critic weights.

        Returns:
            dict: Dictionary containing actor and critic weights.
        """
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }


def train_td3_bc(env, agent, episodes=1000, max_steps=500, log_prefix="td3_bc_exp", model_path=None, pretrain_path=None, logger=None):
    """
    Train a TD3+BC agent on the given environment.

    Args:
        env: Environment instance with reset() and step() methods.
        agent: TD3BCAgent instance.
        episodes (int): Number of episodes to train.
        max_steps (int): Max steps per episode.
        log_prefix (str): Prefix for logging directories and files.
        model_path (str or None): Optional path to save initial model.
        pretrain_path (str or None): Optional path to load pretrained weights.

    Returns:
        List of episode rewards.
    """
    rewards = []
    env.max_steps = max_steps  # Set environment max steps
    global_step = 0  # Global step counter across episodes

    # Setup tensorboard logging directory
    tb_log_dir = f"runs/{log_prefix}"
    # Setup log file path
    log_file_path = f"logs/{log_prefix}.log"
    # Directory for saving models
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # # Setup logger for training information
    logger = logging.getLogger(f"td3_bc_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Clear existing tensorboard logs to start fresh
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Load pretrained model weights if provided
    if pretrain_path and os.path.exists(pretrain_path):
        agent.load(pretrain_path)
        print(f"✅ Successfully loaded pretrained model: {pretrain_path}")
        logger.info(f"Successfully loaded pretrained model: {pretrain_path}")
    else:
        print("❌ No pretrained model loaded, training from scratch")
        logger.info("No pretrained model loaded, training from scratch")

    # Main training loop over episodes
    for ep in tqdm(range(episodes), desc="TD3+BC Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        # Loop for max_steps or until episode done
        for step in range(max_steps):
            action = agent.act(state)  # Select action from agent
            next_state, reward, done, info = env.step(action)  # Take action in env

            # Store transition and perform one training step
            agent.remember(state, action, reward, next_state, float(done))
            actor_loss, critic_loss, q_value = agent.train_step()

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # Log scalar metrics to TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            # Log losses and Q-value if available
            if actor_loss is not None:
                writer.add_scalar("Loss/Actor", actor_loss, global_step)
            if critic_loss is not None:
                writer.add_scalar("Loss/Critic", critic_loss, global_step)
            if q_value is not None:
                writer.add_scalar("Q_value", q_value, global_step)

            # Log training progress to file
            logger.info(
                f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | "
                f"Loss: {{'q1_loss': {critic_loss if critic_loss else 0.0:.2f}, 'policy_loss': {actor_loss if actor_loss else 0.0:.2f}}}"
            )

            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        # Store total episode reward
        rewards.append(ep_reward)
        # Log episode-level rewards to TensorBoard
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        # Log summary info for episode
        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
            f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
        )

        # Save checkpoint every 20 episodes and at the end
        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] Model checkpoint saved at {ckpt_path}")

    # Save final model weights after training completes
    agent.save(final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")
    writer.close()

    return rewards
