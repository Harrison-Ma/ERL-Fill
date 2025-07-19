import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import shutil
import time
import logging
from collections import deque
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from CommonInterface.Logger import init_logger


# === Emotion Replay Buffer ===
class EmotionReplayBuffer:
    """
    Replay buffer that stores experience tuples (transitions) for training
    reinforcement learning agents with emotion-augmented state representations.
    Implements a fixed-size buffer using deque with a maximum capacity.
    """
    def __init__(self, capacity=100000):
        """
        Initialize the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store in the buffer.
        """
        self.buffer = deque(maxlen=capacity)

    def add(self, transition):
        """
        Add a new transition to the buffer.

        Args:
            transition (tuple): A tuple representing one experience (state, action, reward, next_state, done, emotion, etc.).
        """
        self.buffer.append(transition)

    def sample(self, batch_size):
        """
        Randomly sample a batch of transitions from the buffer.

        Args:
            batch_size (int): Number of transitions to sample.

        Returns:
            list: A list of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        """
        Get the current size of the buffer.

        Returns:
            int: Number of transitions currently stored.
        """
        return len(self.buffer)


class EmotionGaussianPolicy(nn.Module):
    """
    Gaussian policy network that incorporates emotional state information.
    Outputs mean and log standard deviation for action distributions,
    modulated by emotional features.

    Args:
        state_dim (int): Dimension of the input state vector.
        action_dim (int): Dimension of the output action vector.
        emotion_dim (int): Dimension of the emotional state vector.
        hidden_size (int): Number of hidden units in fully connected layers.
    """
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256):
        super().__init__()
        # Encoder for state input
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        # Encoder for emotion input
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )

        # Policy network combines encoded state and emotion features
        self.policy_net = nn.Sequential(
            nn.Linear(hidden_size + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU()
        )

        # Output layers for mean and log standard deviation of action distribution
        self.mean_layer = nn.Linear(hidden_size, action_dim)
        self.log_std_layer = nn.Linear(hidden_size, action_dim)

        # Emotion modulation layers to adjust mean and log_std based on emotion input
        self.emotion_adapter_mean = nn.Sequential(
            nn.Linear(emotion_dim, action_dim),
            nn.Tanh()  # Ensures output is in range [-1, 1]
        )

        self.emotion_adapter_std = nn.Sequential(
            nn.Linear(emotion_dim, action_dim),
            nn.Tanh()  # Modulates variance size, output in [-1, 1]
        )

    def forward(self, state, emotion):
        """
        Forward pass through the policy network.

        Args:
            state (Tensor): Batch of states, shape (batch_size, state_dim).
            emotion (Tensor): Batch of emotion vectors, shape (batch_size, emotion_dim).

        Returns:
            Tuple[Tensor, Tensor]: Mean and log standard deviation of the action distribution.
        """
        state_feat = self.state_encoder(state)          # Encode state features
        emo_feat = self.emotion_encoder(emotion)        # Encode emotion features
        x = self.policy_net(torch.cat([state_feat, emo_feat], dim=-1))  # Combine features

        base_mean = self.mean_layer(x)                   # Base mean output
        base_log_std = self.log_std_layer(x)             # Base log std output

        # Emotion modulation
        mean_adjust = self.emotion_adapter_mean(emotion)  # Adjust mean by emotion
        std_adjust = self.emotion_adapter_std(emotion)    # Adjust log std by emotion

        mean = base_mean + mean_adjust
        log_std = base_log_std + std_adjust
        log_std = torch.clamp(log_std, -20, 2)           # Clamp log_std for numerical stability

        return mean, log_std

    def sample(self, state, emotion):
        """
        Sample actions from the policy's Gaussian distribution using reparameterization trick.

        Args:
            state (Tensor): Batch of states.
            emotion (Tensor): Batch of emotions.

        Returns:
            Tuple[Tensor, Tensor]: Sampled actions after tanh squashing, and corresponding log probabilities.
        """
        mean, log_std = self(state, emotion)
        std = torch.exp(log_std)

        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # Re parameterization trick for backpropagation
        action = torch.tanh(z)  # Squash action to range [-1, 1]

        # Compute log probability with correction for tanh squashing
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        return action, log_prob.sum(dim=-1, keepdim=True)

    def get_action_entropy(self, state, emotion):
        """
        Calculate the entropy of the action distribution given state and emotion.

        Args:
            state (Tensor): Batch of states.
            emotion (Tensor): Batch of emotions.

        Returns:
            float: Mean entropy across the batch.
        """
        mean, log_std = self(state, emotion)
        # Per-dimension entropy of Gaussian: log_std + 0.5 * log(2*pi*e)
        entropy = log_std + 0.5 * np.log(2 * np.pi * np.e)
        return entropy.sum(dim=-1).mean().item()  # Sum across dimensions, average over batch


# === Emotion-Aware Q Network ===
class EmotionAwareQNetwork(nn.Module):
    """
    Q-network that incorporates emotional state information along with state and action inputs.
    Estimates the Q-value for given (state, action, emotion) tuples.

    Args:
        state_dim (int): Dimension of state input.
        action_dim (int): Dimension of action input.
        emotion_dim (int): Dimension of emotion input.
        hidden_size (int): Number of hidden units in each layer.
    """

    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256):
        super().__init__()
        # Encoder for state input
        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        # Encoder for action input
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU()
        )
        # Encoder for emotion input
        self.emotion_encoder = nn.Sequential(
            nn.Linear(emotion_dim, 32),
            nn.LayerNorm(32),
            nn.ReLU()
        )
        # Combined Q-network that outputs scalar Q-value
        self.q_net = nn.Sequential(
            nn.Linear(hidden_size * 2 + 32, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, state, action, emotion):
        """
        Forward pass to compute Q-value given state, action, and emotion.

        Args:
            state (Tensor): Batch of states.
            action (Tensor): Batch of actions.
            emotion (Tensor): Batch of emotions.

        Returns:
            Tensor: Q-values, shape (batch_size, 1).
        """
        s_feat = self.state_encoder(state)
        a_feat = self.action_encoder(action)
        e_feat = self.emotion_encoder(emotion)
        x = torch.cat([s_feat, a_feat, e_feat], dim=-1)
        return self.q_net(x)


# === EmotionSAC Agent ===
class EmotionSACAgent:
    """
    Soft Actor-Critic (SAC) agent enhanced with emotion awareness.
    Integrates emotional information into policy and Q-value estimations,
    and applies emotional regularization during policy updates.

    Args:
        env: Environment instance providing state/action specs and bounds.
        lambda_emo (float): Weight for emotion regularization term in policy loss.
        device (str): Device to run the model on ('cpu' or 'cuda').
    """

    def __init__(self, env, lambda_emo=0.05, device='cuda'):
        self.env = env
        self.device = device
        self.lambda_emo = lambda_emo

        # Environment specs
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.gamma = 0.99  # Discount factor
        self.tau = 0.005  # Target network update rate
        self.alpha = 0.3  # Entropy regularization coefficient
        self.batch_size = 128  # Batch size for training

        self.train_step = 1

        # Import emotion module utilities
        from EmotionModule import EmotionModule, save_emotion_module, load_emotion_module
        self.emotion = EmotionModule(device=device)
        self.save_emotion_module = save_emotion_module
        self.load_emotion_module = load_emotion_module

        # Initialize networks
        self.policy = EmotionGaussianPolicy(self.state_dim, self.action_dim).to(device)
        self.q1 = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q2 = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q1_target = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)
        self.q2_target = EmotionAwareQNetwork(self.state_dim, self.action_dim).to(device)

        # Copy weights to target networks
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Optimizers for networks
        self.policy_optim = optim.Adam(self.policy.parameters(), lr=3e-4)
        self.q1_optim = optim.Adam(self.q1.parameters(), lr=3e-4)
        self.q2_optim = optim.Adam(self.q2.parameters(), lr=3e-4)

        # Replay buffer
        self.memory = EmotionReplayBuffer()

        # Calculate action normalization offsets and scales from env bounds
        self.offsets = np.array([(hi + lo) / 2 for lo, hi in env.bounds.values()])
        self.scales = np.array([(hi - lo) / 2 for lo, hi in env.bounds.values()])

    def normalize_action(self, action):
        """
        Normalize raw action to range [-1, 1] using environment bounds.

        Args:
            action (np.ndarray): Raw action values.

        Returns:
            np.ndarray: Normalized action.
        """
        return (action - self.offsets) / self.scales

    def denormalize_action(self, norm_action):
        """
        Convert normalized action back to raw action space.

        Args:
            norm_action (np.ndarray): Normalized action values.

        Returns:
            np.ndarray: Denormalized action.
        """
        return norm_action * self.scales + self.offsets

    def act(self, state, deterministic=False, add_noise=True):
        """
        Select an action given the current state and emotion.

        Args:
            state (np.ndarray): Current environment state.
            deterministic (bool): Whether to use deterministic policy (mean action).
            add_noise (bool): Whether to sample stochastically from policy.

        Returns:
            np.ndarray: Action in raw environment action space.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        emotion = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)

        with torch.no_grad():
            if deterministic:
                mean, _ = self.policy(state_tensor, emotion)
                action = torch.tanh(mean)  # Deterministic action
            elif add_noise:
                action, _ = self.policy.sample(state_tensor, emotion)  # Stochastic action with noise
            else:
                mean, _ = self.policy(state_tensor, emotion)
                action = torch.tanh(mean)  # Deterministic without noise

        action = action.cpu().numpy().squeeze()
        return self.denormalize_action(action)

    def remember(self, s, a, r, s_, d):
        """
        Store a transition in the replay buffer.

        Args:
            s (np.ndarray): Current state.
            a (np.ndarray): Action taken.
            r (float): Reward received.
            s_ (np.ndarray): Next state.
            d (bool): Done flag indicating episode termination.
        """
        norm_a = self.normalize_action(a)
        self.memory.add((s, norm_a, r, s_, d, self.emotion.get_emotion()))

    def update(self):
        """
        Perform one training update step for the SAC agent.

        Returns:
            dict or None: Training losses and emotion regularization metric if update performed, else None.
        """
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch from replay buffer
        batch = self.memory.sample(self.batch_size)
        s, a, r, s_, d, e = zip(*batch)

        # Convert batch data to tensors on device
        s = torch.FloatTensor(s).to(self.device)
        a = torch.FloatTensor(a).to(self.device)
        r = torch.FloatTensor(r).unsqueeze(1).to(self.device)
        s_ = torch.FloatTensor(s_).to(self.device)
        d = torch.FloatTensor(d).unsqueeze(1).to(self.device)
        e = torch.FloatTensor(e).to(self.device)

        # Compute target Q-values using target networks and next actions
        with torch.no_grad():
            next_action, next_log_pi = self.policy.sample(s_, e)
            target_q1 = self.q1_target(s_, next_action, e)
            target_q2 = self.q2_target(s_, next_action, e)
            min_target_q = torch.min(target_q1, target_q2) - self.alpha * next_log_pi
            target_q = r + (1 - d) * self.gamma * min_target_q

        # Compute current Q estimates and losses
        current_q1 = self.q1(s, a, e)
        current_q2 = self.q2(s, a, e)
        q1_loss = nn.MSELoss()(current_q1, target_q)
        q2_loss = nn.MSELoss()(current_q2, target_q)

        # Optimize Q1 network
        self.q1_optim.zero_grad()
        q1_loss.backward()
        self.q1_optim.step()

        # Optimize Q2 network
        self.q2_optim.zero_grad()
        q2_loss.backward()
        self.q2_optim.step()

        # Update policy network
        new_action, log_pi = self.policy.sample(s, e)
        q1_new = self.q1(s, new_action, e)
        policy_loss = (self.alpha * log_pi - q1_new).mean()

        # === Emotion regularization ===
        # Retrieve current and previous emotions as tensors
        e_now = self.emotion.get_emotion()  # numpy array, shape=(3,)
        if not hasattr(self, 'prev_emotion'):
            self.prev_emotion = e_now  # Initialize previous emotion if first update

        e_now_tensor = torch.FloatTensor(e_now).to(self.device)
        e_prev_tensor = torch.FloatTensor(self.prev_emotion).to(self.device)
        emo_reg = torch.norm(e_now_tensor - e_prev_tensor, p=2)  # L2 norm between current and previous emotion

        # Total policy loss includes emotion regularization weighted by lambda_emo
        total_policy_loss = policy_loss + self.lambda_emo * emo_reg
        print("self.lambda_emo:###############", self.lambda_emo)

        # Update previous emotion for next iteration
        self.prev_emotion = e_now

        # Optimize policy network with total loss
        self.policy_optim.zero_grad()
        total_policy_loss.backward()
        self.policy_optim.step()

        # Soft update target networks
        for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "q1_loss": q1_loss.item(),
            "q2_loss": q2_loss.item(),
            "policy_loss": policy_loss.item(),
            "emo_reg": emo_reg.item()  # Can be logged to TensorBoard for monitoring
        }

    def save(self, path):
        """
        Save policy model and emotion module (if available) to disk.

        Args:
            path (str): File path for saving policy weights.
        """

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict()
        }, path)

        # Save emotion module weights if emotion transformer exists
        if hasattr(self.emotion, 'transformer'):
            self.save_emotion_module(self.emotion, path.replace(".pth", "_emo.pth"))

    def load(self, path):
        """
        Load policy model and emotion module (if available) from disk.

        Args:
            path (str): File path to load policy weights from.
        """
        checkpoint = torch.load(path)
        self.policy.load_state_dict(checkpoint["policy"])
        self.policy.eval()

        # Load emotion module weights if transformer exists
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


# # Logging configuration
# logger = logging.getLogger(__name__)
# os.makedirs("logs", exist_ok=True)
# logging.basicConfig(filename="logs/sac_training_transformer.log", filemode="w",
#                     level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def train_erl_fill(env, agent, episodes=1000, max_steps=500, log_prefix="emotion_sac", model_path=None, logger=None,
                   lambda_emo=0.05):
    """
    Train an emotion-augmented Soft Actor-Critic (SAC) agent on a given environment.
    """
    if logger is None:
        logger = init_logger(log_prefix=log_prefix)

    agent.lambda_emo = lambda_emo
    if logger: logger.info(f"λ_emo set to {agent.lambda_emo:.3f}")

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

            agent.emotion.update(reward, action)

            agent.remember(state, action, reward, next_state, float(done))
            update_info = agent.update()

            state = next_state
            ep_reward += reward
            ep_err += info.get("weight_error", 0.0)
            ep_time += info.get("total_time", 0.0)
            average_step += 1

            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)
            writer.add_scalar("Metric/SlowWeight", info.get("action", {}).get("slow_weight", 0.0), global_step)

            emotion_state = agent.emotion.get_emotion()
            writer.add_scalar("Emotion/Curiosity", float(emotion_state[0]), global_step)
            writer.add_scalar("Emotion/Conservativeness", float(emotion_state[1]), global_step)
            writer.add_scalar("Emotion/Anxiety", float(emotion_state[2]), global_step)

            if update_info:
                writer.add_scalar("Loss/Actor", update_info.get("policy_loss", 0.0), global_step)
                writer.add_scalar("Loss/Critic", update_info.get("q1_loss", 0.0), global_step)

            if step % (max_steps // 10) == 0 or step == max_steps - 1:
                if logger:
                    logger.info(
                        f"Ep {ep:03d} Step {step:03d} | R: {reward:.2f} | Loss: {update_info} | "
                        f"E: {np.round(emotion_state, 2)}"
                    )

            global_step += 1
            if done:
                if logger:
                    logger.info(f"✔️ Episode {ep} finished early at step {step}")
                break

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)
        writer.add_scalar("Metric/Weight", info.get("final_weight", 0.0), ep)

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
        if logger:
            logger.info(
                f"[Episode {ep:04d}] "
                f"TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
                f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
                f"Emotion: [Curi: {emotion_state[0]:.2f}, Cons: {emotion_state[1]:.2f}, Anx: {emotion_state[2]:.2f}] | "
                f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
            )

        if ep > 0 and ep % 100 == 0:
            duration_100 = time.time() - start_100
            if logger:
                logger.info(
                    f"⏱️ Episodes {ep - 100}~{ep} took {duration_100:.2f} seconds, "
                    f"avg {duration_100 / 100:.2f} seconds/episode"
                )
            start_100 = time.time()

        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(saved_model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            if logger:
                logger.info(f"[Episode {ep}] Model checkpoint saved at {ckpt_path}")

        agent.train_step += 1

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(agent.device)
            emotion_tensor = torch.FloatTensor(agent.emotion.get_emotion()).unsqueeze(0).to(agent.device)
            action_entropy = agent.policy.get_action_entropy(state_tensor, emotion_tensor)

        writer.add_scalar("Metrics/AvgReward", ep_reward / average_step, ep)
        writer.add_scalar("Metrics/AvgError", ep_err / average_step, ep)
        writer.add_scalar("Metrics/CompletionTime", ep_time / average_step, ep)
        reward_variance = np.var(rewards[-100:] if len(rewards) >= 100 else rewards)
        writer.add_scalar("Metrics/RewardVariance", reward_variance, ep)
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
    if logger: logger.info(f"✅ Final model saved at {model_path}")
    writer.close()

    import pandas as pd
    metrics_df = pd.DataFrame(metrics_summary)
    metrics_df.to_csv(f"{saved_model_dir}/{log_prefix}_metrics_summary_{lambda_emo}.csv", index=False)

    return rewards

