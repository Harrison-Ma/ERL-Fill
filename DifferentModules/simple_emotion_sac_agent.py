import torch.nn as nn
import torch.optim as optim
import random
import os
import shutil
import torch
import numpy as np
import logging
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from collections import deque


class EmotionModuleSimple:
    """
    A simple emotion modeling module that tracks three emotional dimensions:
    exploration, conservativeness, and anxiety. Emotion is updated gradually
    based on the agent's reward signal and movement magnitude.
    """
    def __init__(self):
        # Initial emotional state: [exploration, conservativeness, anxiety]
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.alpha = 0.02  # Slow update rate for smooth emotional transitions

    def update(self, reward, movement):
        """
        Update the emotional state based on the current reward and movement vector.

        Args:
            reward (float): The reward signal from the environment.
            movement (np.ndarray): The agent's action or motion vector.
        """
        reward_term = 1 - np.tanh(reward)  # Encourages exploration when reward is low
        move_mag = np.clip(np.linalg.norm(movement), 0.0, 1.0)  # Normalized movement magnitude

        exploration = 0.5 * reward_term + 0.5 * move_mag
        conservativeness = np.clip(np.tanh(reward) - 0.1 * move_mag, 0.0, 1.0)
        anxiety = np.clip(self.current_emotion[2] - 0.01 * np.tanh(reward), 0.0, 1.0)

        new_emotion = (1 - self.alpha) * self.current_emotion + self.alpha * np.array([
            exploration, conservativeness, anxiety
        ])
        self.current_emotion = np.clip(new_emotion, 0.0, 1.0)

    def get_emotion(self):
        """
        Return a copy of the current emotional state.

        Returns:
            np.ndarray: A 3-element array representing the emotional state.
        """
        return self.current_emotion.copy()


class GaussianPolicy(nn.Module):
    """
    Gaussian policy network for continuous action spaces.
    Takes both state and emotional state as input.
    """
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_dim=256):
        """
        Initialize the Gaussian policy network.

        Args:
            state_dim (int): Dimension of the environment state.
            action_dim (int): Dimension of the action space.
            emotion_dim (int): Dimension of the emotion vector (default: 3).
            hidden_dim (int): Size of hidden layers.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + emotion_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU()
        )
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.log_std = nn.Linear(hidden_dim, action_dim)

    def forward(self, state, emotion):
        """
        Forward pass through the policy network.

        Args:
            state (torch.Tensor): Batch of environment states.
            emotion (torch.Tensor): Batch of emotion vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Mean and standard deviation of the action distribution.
        """
        x = torch.cat([state, emotion], dim=-1)
        x = self.net(x)
        mean = self.mean(x)
        log_std = self.log_std(x).clamp(-20, 2)
        std = log_std.exp()
        return mean, std

    def sample(self, state, emotion):
        """
        Sample an action from the policy using the reparameterization trick.

        Args:
            state (torch.Tensor): Batch of environment states.
            emotion (torch.Tensor): Batch of emotion vectors.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Sampled action and its log probability.
        """
        mean, std = self(state, emotion)
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Reparameterized sample
        action = torch.tanh(z)
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-7)
        return action, log_prob.sum(dim=-1, keepdim=True)


class QNetwork(nn.Module):
    """
    Q-network for estimating the value of state-action pairs.
    """
    def __init__(self, state_dim, action_dim, hidden_dim=256):
        """
        Initialize the Q-network.

        Args:
            state_dim (int): Dimension of the environment state.
            action_dim (int): Dimension of the action space.
            hidden_dim (int): Size of hidden layers.
        """
        super().__init__()
        self.q = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, state, action):
        """
        Compute Q-value for given state and action.

        Args:
            state (torch.Tensor): Batch of environment states.
            action (torch.Tensor): Batch of actions.

        Returns:
            torch.Tensor: Estimated Q-values.
        """
        return self.q(torch.cat([state, action], dim=-1))


class MemoryWrapper:
    """
    A simple experience replay buffer using deque.
    Stores past transitions for sampling during training.
    """
    def __init__(self, maxlen=100000):
        """
        Initialize the memory buffer.

        Args:
            maxlen (int): Maximum number of items to store.
        """
        self.buffer = deque(maxlen=maxlen)

    def append(self, item):
        """
        Append a new experience to the buffer.

        Args:
            item (tuple): A transition tuple (s, a, r, s_, done).
        """
        self.buffer.append(item)

    def __len__(self):
        """
        Return the current size of the buffer.
        """
        return len(self.buffer)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of samples to return.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)


class SimpleEmotionSACAgent:
    """
    Soft Actor-Critic (SAC) agent with a simple emotion integration.
    Emotion is used as an additional input to the policy network.
    """
    def __init__(self, env, device='cuda'):
        """
        Initialize the agent and neural networks.

        Args:
            env: The environment with state/action dimensions and bounds.
            device (str): PyTorch device to run the networks on.
        """
        self.env = env
        self.device = device
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005   # Target smoothing coefficient
        self.batch_size = 128
        self.alpha = 0.3   # Entropy coefficient (fixed)

        # Emotion module (affects policy decisions)
        self.emotion = EmotionModuleSimple()

        # Actor and critic networks
        self.policy = GaussianPolicy(env.state_dim, env.action_dim).to(device)
        self.q1 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2 = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q1_target = QNetwork(env.state_dim, env.action_dim).to(device)
        self.q2_target = QNetwork(env.state_dim, env.action_dim).to(device)

        # Copy initial parameters to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        # Action normalization utilities
        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

        # Replay memory
        self.memory = MemoryWrapper(maxlen=100000)

    def denormalize_action(self, norm_action):
        """
        Convert normalized action back to the environment's original scale.
        """
        return norm_action * self.scales + self.offsets

    def normalize_action(self, real_action):
        """
        Normalize action to [-1, 1] range.
        """
        return (real_action - self.offsets) / self.scales

    def act(self, state, deterministic=False):
        """
        Select an action given the current state and emotional state.

        Args:
            state (np.ndarray): Environment state.
            deterministic (bool): Whether to return the mean action or sample.

        Returns:
            np.ndarray: Denormalized action.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        emotion_tensor = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)
        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_tensor, emotion_tensor)
                action = torch.tanh(mean).cpu().numpy()[0]
            else:
                action, _ = self.policy.sample(state_tensor, emotion_tensor)
                action = action.cpu().numpy()[0]
        return self.denormalize_action(action)

    def remember(self, s, a, r, s_, d):
        """
        Store a transition in the replay buffer.

        Args:
            s (np.ndarray): State.
            a (np.ndarray): Action.
            r (float): Reward.
            s_ (np.ndarray): Next state.
            d (bool): Done flag.
        """
        norm_a = self.normalize_action(a)
        self.memory.append((s, norm_a, r, s_, d))

    def update(self):
        """
        Sample a batch of experiences and update networks using SAC loss.

        Returns:
            dict: Training losses for Q1, Q2, and the policy.
        """
        if len(self.memory) < self.batch_size:
            return  # Not enough samples yet

        # Sample a mini-batch of transitions
        batch = self.memory.sample(self.batch_size)
        s, a, r, s_, d = zip(*batch)

        # Convert to tensors
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)

        # Get current emotional state for batch
        e = torch.FloatTensor([self.emotion.get_emotion() for _ in range(self.batch_size)]).to(self.device)

        # === Critic Update ===
        with torch.no_grad():
            next_action, log_pi_next = self.policy.sample(s_, e)
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

        # === Policy Update ===
        new_action, log_pi = self.policy.sample(s, e)
        q1_new = self.q1(s, new_action)
        policy_loss = (self.alpha * log_pi - q1_new).mean()

        self.policy_optim.zero_grad()
        policy_loss.backward()
        self.policy_optim.step()

        # === Target Network Update ===
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
        Save the policy network parameters to disk.

        Args:
            path (str): File path to save the policy.
        """
        torch.save({"policy": self.policy.state_dict()}, path)

    def load(self, path):
        """
        Load policy network parameters from disk.

        Args:
            path (str): File path from which to load the policy.
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()


def train_simple_sac(env, agent, episodes=1000, max_steps=500, log_prefix="simple_sac_exp", model_path=None, pretrain_path=None):
    """
    Trains a Soft Actor-Critic (SAC) agent with an optional emotional module in a given environment.

    Args:
        env: The environment to train the agent in. Must follow OpenAI Gym interface.
        agent: The SAC agent to be trained. Must implement `act`, `remember`, `update`, `save`, and `load`.
        episodes (int): Number of training episodes.
        max_steps (int): Maximum number of steps per episode.
        log_prefix (str): Prefix used for logging directory and saved model naming.
        model_path (str): Optional path for saving initial model checkpoint.
        pretrain_path (str): Optional path for loading pretrained weights.

    Returns:
        List of cumulative rewards per episode.
    """
    rewards = []
    env.max_steps = max_steps  # Ensure environment is aware of the max episode length
    global_step = 0  # Global step counter across all episodes

    # === Setup logging and output paths ===
    tb_log_dir = f"runs/{log_prefix}"  # Directory for TensorBoard logs
    log_file_path = f"logs/sac_training_{log_prefix}.log"  # File path for logging training progress
    model_dir = os.path.join("saved_models", log_prefix)  # Directory to save model checkpoints
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # === Configure logger ===
    logger = logging.getLogger(f"sac_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Remove previous TensorBoard logs if they exist
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # === Load pretrained model if specified ===
    if pretrain_path and os.path.exists(pretrain_path):
        print("pretrain_path:", pretrain_path)
        agent.load(pretrain_path)
        print(f"✅ Loaded pretrained model from: {pretrain_path}")
        logger.info(f"Loaded pretrained model from: {pretrain_path}")
    else:
        print("❌ No pretrained model found, training from scratch")
        logger.info("No pretrained model found, training from scratch")

    # === Main training loop ===
    for ep in tqdm(range(episodes), desc="Simple SAC Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state)  # Select action using policy
            next_state, reward, done, info = env.step(action)  # Interact with environment

            # Store transition in memory and update networks
            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()

            # Emotion module update (if implemented)
            if hasattr(agent, "emotion") and hasattr(agent.emotion, "update"):
                agent.emotion.update(reward, action)

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # Get current emotion state
            emotion = agent.emotion.get_emotion()

            # === TensorBoard logging ===
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

            # End episode early if done
            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        # === End of episode logging ===
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

        # Periodically save checkpoint
        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] Model checkpoint saved at {ckpt_path}")

    # === Save final model ===
    agent.save(final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")
    writer.close()

    return rewards
