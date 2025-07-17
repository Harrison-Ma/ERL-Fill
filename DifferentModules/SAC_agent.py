import torch.nn as nn
import torch.optim as optim
import random
from collections import deque
from EmotionModule import EmotionModuleNone
import os
import shutil
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import logging


class GaussianPolicy(nn.Module):
    """
    GaussianPolicy represents a stochastic policy using a Gaussian distribution.
    It outputs the mean and standard deviation for each action dimension, enabling sampling with exploration.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Gaussian policy network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the output action.
            hidden_dim (int): Number of hidden units per layer.
        """
        super().__init__()
        # Feature extraction MLP
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        # Linear layer to produce mean of the Gaussian
        self.mean = nn.Linear(hidden_dim, action_dim)
        # Linear layer to produce log of the standard deviation
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        """
        Forward pass to compute the mean and standard deviation of the action distribution.

        Args:
            state (Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            Tuple[Tensor, Tensor]: Mean and standard deviation of the Gaussian action distribution.
        """
        x = self.net(state)
        mean = self.mean(x)
        # Clamp log_std to prevent numerical instability
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()  # Ensure std is positive
        return mean, std

    def sample(self, state):
        """
        Sample an action using the reparameterization trick and apply tanh squashing.

        Args:
            state (Tensor): Input state tensor of shape (batch_size, state_dim).

        Returns:
            Tuple[Tensor, Tensor]: Tanh-squashed action and log-probability with correction term.
        """
        mean, std = self(state)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterization trick
        action = torch.tanh(z)  # Bound action to (-1, 1)

        # Apply correction for tanh squashing to the log-probability
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        log_prob = log_prob.sum(dim=-1, keepdim=True)  # Sum over action dimensions
        return action, log_prob


class QNetwork(nn.Module):
    """
    QNetwork estimates the action-value function Q(s, a) using a feedforward neural network.
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Q-network.

        Args:
            state_dim (int): Dimension of the input state.
            action_dim (int): Dimension of the input action.
            hidden_dim (int): Number of hidden units per layer.
        """
        super().__init__()
        # Concatenate state and action, then feed into MLP to estimate Q-value
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Forward pass to compute Q(s, a).

        Args:
            state (Tensor): State tensor of shape (batch_size, state_dim).
            action (Tensor): Action tensor of shape (batch_size, action_dim).

        Returns:
            Tensor: Estimated Q-value of shape (batch_size, 1).
        """
        x = torch.cat([state, action], dim=-1)  # Concatenate state and action
        return self.q(x)


class SACAgent:
    """
    Soft Actor-Critic (SAC) agent implementation for continuous control tasks.
    """

    def __init__(self, env, device='cuda'):
        """
        Initialize the SAC agent, including policy and Q-networks, optimizers, and replay buffer.

        Args:
            env: The environment with defined state/action space and bounds.
            device (str): Device for computation, either 'cuda' or 'cpu'.
        """
        self.env = env
        self.device = device
        self.gamma = 0.99                # Discount factor
        self.tau = 0.005                 # Target network update rate
        self.batch_size = 128            # Training batch size
        self.alpha = 0.3                 # Entropy regularization coefficient (fixed)

        # Initialize policy and Q-networks
        self.policy = GaussianPolicy(env.state_dim, env.action_dim).to(device)
        self.q1 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q1_target = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2_target = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Set up optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        # Precompute scaling and offset for action normalization
        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

        # Replay buffer (simple deque)
        self.memory = type('DummyMemory', (), {
            'buffer': deque(maxlen=100000),
            '__len__': lambda self: len(self.buffer)
        })()

        # Emotion module hook (optional, currently a placeholder)
        self.emotion_module = EmotionModuleNone()

    def denormalize_action(self, norm_action):
        """
        Convert normalized action back to original action space.

        Args:
            norm_action (np.ndarray): Normalized action in [-1, 1].

        Returns:
            np.ndarray: Real-valued action in environment-specific range.
        """
        return norm_action * self.scales + self.offsets

    def normalize_action(self, real_action):
        """
        Normalize real-valued action to [-1, 1] for training stability.

        Args:
            real_action (np.ndarray): Original action.

        Returns:
            np.ndarray: Normalized action.
        """
        return (real_action - self.offsets) / self.scales

    def act(self, state, deterministic=False):
        """
        Select an action given the current state using the policy network.

        Args:
            state (np.ndarray): Current environment state.
            deterministic (bool): Whether to use deterministic action (mean) or sample.

        Returns:
            np.ndarray: Action in real-world scale.
        """
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
        """
        Store transition in replay buffer after normalizing the action.

        Args:
            s (np.ndarray): Current state.
            a (np.ndarray): Action taken.
            r (float): Reward received.
            s_ (np.ndarray): Next state.
            d (bool): Done flag indicating episode termination.
        """
        norm_a = self.normalize_action(a)
        self.memory.buffer.append((s, norm_a, r, s_, d))

    def update(self):
        """
        Perform a single training step using a batch from the replay buffer.

        Returns:
            dict: Dictionary of loss values for logging.
        """
        if len(self.memory.buffer) < self.batch_size:
            return

        # Sample mini-batch
        batch = random.sample(self.memory.buffer, self.batch_size)
        s, a, r, s_, d = zip(*batch)
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # === Update Critic networks ===
        with torch.no_grad():
            next_action, log_pi_next = self.policy.sample(s_)
            q1_next = self.q1_target(s_, next_action)
            q2_next = self.q2_target(s_, next_action)
            min_q_next = torch.min(q1_next, q2_next)
            target_q = r + (1 - d) * self.gamma * (min_q_next - self.alpha * log_pi_next)

        # Compute current Q estimates
        q1 = self.q1(s, a)
        q2 = self.q2(s, a)
        q1_loss = nn.MSELoss()(q1, target_q)
        q2_loss = nn.MSELoss()(q2, target_q)

        # Optimize Q-networks
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # === Update Policy network ===
        new_action, log_pi = self.policy.sample(s)
        q1_new = self.q1(s, new_action)
        policy_loss = (self.alpha * log_pi - q1_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # === Soft update target networks ===
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
        """
        Save the policy model to a file.

        Args:
            path (str): File path to save the model.
        """
        torch.save({
            "policy": self.policy.state_dict()
        }, path)

    def load(self, path):
        """
        Load the policy model from a file.

        Args:
            path (str): File path from which to load the model.
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()


# Logging configuration
logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    filename="logs/sac_training.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def train_sac(env, agent, episodes=1000, max_steps=500, log_prefix="simple_sac_exp",
              model_path=None, pretrain_path=None):
    """
    Train a Soft Actor-Critic (SAC) agent in a given environment.

    Args:
        env: The environment to train on.
        agent: An instance of SACAgent.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum steps per episode.
        log_prefix (str): Prefix used for logs and model paths.
        model_path (str or None): Optional path to save intermediate models.
        pretrain_path (str or None): Optional path to load a pretrained model.

    Returns:
        list: Episode reward for each training episode.
    """
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # Set up log and model directories
    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/sac_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)
    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")

    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # Logger setup
    logger = logging.getLogger(f"sac_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Remove existing TensorBoard logs
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Optionally load pretrained model
    if pretrain_path and os.path.exists(pretrain_path):
        print("pretrain_path:", pretrain_path)
        agent.load(pretrain_path)
        print(f"✅ Loaded pretrained model: {pretrain_path}")
        logger.info(f"Loaded pretrained model: {pretrain_path}")
    else:
        print("❌ No pretrained model found. Training from scratch.")
        logger.info("No pretrained model found. Training from scratch.")

    for ep in tqdm(range(episodes), desc="Simple SAC Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)

            # Store transition and update agent
            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # Get emotion from the agent's emotion module (if available)
            if hasattr(agent, "emotion_module"):
                emotion = agent.emotion_module.get_emotion()
            else:
                emotion = [0.5, 0.5, 0.0]  # Default neutral emotion

            # Log metrics to TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            if update_info:
                writer.add_scalar("Loss/Actor", update_info.get("policy_loss", 0.0), global_step)
                writer.add_scalar("Loss/Critic", update_info.get("q1_loss", 0.0), global_step)
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | "
                    f"Loss: {{'q1_loss': {update_info.get('q1_loss', 0.0):.2f}, "
                    f"'policy_loss': {update_info.get('policy_loss', 0.0):.2f}}} | "
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

        # Get final emotion state at end of episode
        if hasattr(agent, "emotion_module"):
            emotion = agent.emotion_module.get_emotion()
        else:
            emotion = [0.5, 0.5, 0.0]

        # Log rewards and summary
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

        # Save model periodically
        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] Model saved to {ckpt_path}")

    # Save final model
    agent.save(final_model_path)
    logger.info(f"✅ Final model saved to {final_model_path}")
    writer.close()

    return rewards

