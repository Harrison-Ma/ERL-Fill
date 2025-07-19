import numpy as np
import random
import time
import torch
import torch.nn as nn
import torch.optim as optim
import os
import shutil
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from CommonInterface.Logger import init_logger
from EmotionModule import EmotionModuleNone


class Actor(nn.Module):
    """
    Actor network for policy generation with emotion modulation.

    This neural network takes in the current state and an emotion vector to produce actions
    within a specified range, dynamically modulated by emotion-driven perturbations.

    Args:
        state_dim (int): Dimension of the input state vector.
        action_dim (int): Dimension of the output action vector.
        param_bounds (dict): Dictionary of action parameter bounds, mapping each action name to a (low, high) tuple.
        emotion_dim (int): Dimension of the emotion vector. Default is 3 (e.g., curiosity, conservatism, anxiety).
        hidden_size (int): Size of the hidden layers. Default is 128.
        dropout_rate (float): Dropout rate used in the network to prevent overfitting. Default is 0.2.
    """
    def __init__(self, state_dim, action_dim, param_bounds, emotion_dim=3, hidden_size=128, dropout_rate=0.2):
        super(Actor, self).__init__()
        self.dropout_rate = dropout_rate

        # Input layer: concatenates state and emotion
        self.input_layer = nn.Linear(state_dim + emotion_dim, hidden_size)

        # Deep neural network layers
        self.hidden = nn.Sequential(
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, action_dim),
            nn.Softsign()  # Ensures output lies in (-1, 1)
        )

        # Learnable weights that control how emotions affect each action
        self.emotion_weights = nn.Parameter(torch.randn(emotion_dim, action_dim) * 0.2)

        # Registers scaling and offset for action output transformation
        self._init_param_scaling(param_bounds)

    def _init_param_scaling(self, param_bounds):
        """
        Initializes action scaling and offset buffers to map normalized output to real-world action bounds.

        Args:
            param_bounds (dict): Dictionary mapping action names to (low, high) bounds.
        """
        scale_params, offset_params = [], []
        for key in param_bounds.keys():
            low, high = param_bounds[key]
            scale = (high - low) / 2.0
            offset = (high + low) / 2.0
            scale_params.append(scale)
            offset_params.append(offset)
        self.register_buffer('scale_params', torch.tensor(scale_params, dtype=torch.float32))
        self.register_buffer('offset_params', torch.tensor(offset_params, dtype=torch.float32))

    def forward(self, state, emotion):
        """
        Forward pass of the actor network.

        Args:
            state (Tensor): Input state tensor of shape (batch_size, state_dim).
            emotion (Tensor): Emotion vector of shape (batch_size, emotion_dim).

        Returns:
            Tensor: Scaled action tensor of shape (batch_size, action_dim).
        """
        # Ensure proper dimensions
        if emotion.dim() == 1:
            emotion = emotion.unsqueeze(0)
        if state.dim() == 1:
            state = state.unsqueeze(0)
        if emotion.size(0) != state.size(0):
            emotion = emotion.expand(state.size(0), -1)

        # Concatenate state and emotion vectors
        x = torch.cat([state, emotion], dim=1)
        x = self.input_layer(x)
        base_action = self.hidden(x)  # Output range: (-1, 1)

        # === Emotion decomposition ===
        curiosity = emotion[:, 0:1]  # Encourages exploration
        conserv   = emotion[:, 1:2]  # Encourages caution
        anxiety   = emotion[:, 2:3]  # Affects intensity of perturbation

        # === Raw emotion-to-action influence ===
        raw_effect = torch.tanh((emotion @ self.emotion_weights) * 1.5)  # Normalized to (-1, 1)

        # === Dynamic scaling factor (alpha) based on emotions ===
        alpha = 0.1 + 0.9 * anxiety                # Scales with anxiety
        alpha *= (1.0 - 0.5 * conserv)             # Reduced by conservatism
        alpha *= (1.0 + 0.5 * curiosity)           # Boosted by curiosity
        alpha = torch.clamp(alpha, min=0.01, max=1)  # Clamped to avoid instability

        # === Directional modulation to avoid saturation ===
        modulator = (1 - torch.abs(base_action)) ** 2
        sign_flip = -torch.sign(base_action)  # Pushes actions away from saturation bounds
        adjusted_emotion_effect = alpha * raw_effect * sign_flip * modulator  # Emotion-driven noise

        # === Final action computation ===
        raw_action = base_action + adjusted_emotion_effect
        scale_params = self.scale_params.to(raw_action.device)
        offset_params = self.offset_params.to(raw_action.device)
        scaled_action = raw_action * scale_params + offset_params

        return scaled_action


class Critic(nn.Module):
    """
    Critic network for Q-value estimation with emotion influence.

    This network takes a state, action, and emotion vector as input and outputs a scalar Q-value,
    which represents the expected future reward. Emotion input is normalized to stabilize learning.

    Args:
        state_dim (int): Dimension of the state vector.
        action_dim (int): Dimension of the action vector.
        emotion_dim (int): Dimension of the emotion vector (default: 3).
        hidden_size (int): Size of hidden layers (default: 256).
        dropout_rate (float): Dropout rate to prevent overfitting (default: 0.2).
    """
    def __init__(self, state_dim, action_dim, emotion_dim=3, hidden_size=256, dropout_rate=0.2):
        super(Critic, self).__init__()

        # Total input dimension includes state, action, and emotion
        input_dim = state_dim + action_dim + emotion_dim

        # Define the critic network
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_size),
            nn.LayerNorm(hidden_size),       # Helps stabilize training
            nn.LeakyReLU(0.1),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout_rate),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),

            nn.Linear(hidden_size // 2, 1)    # Output: Scalar Q-value
        )

    def forward(self, state, action, emotion):
        """
        Forward pass of the critic network.

        Args:
            state (Tensor): State tensor of shape (batch_size, state_dim).
            action (Tensor): Action tensor of shape (batch_size, action_dim).
            emotion (Tensor): Emotion tensor of shape (batch_size, emotion_dim).

        Returns:
            Tensor: Clamped Q-value tensor of shape (batch_size, 1).
        """
        # Normalize emotion to reduce emotional drift instability
        emotion = (emotion - emotion.mean(dim=0, keepdim=True)) / (emotion.std(dim=0, keepdim=True) + 1e-6)

        # Concatenate inputs and pass through the network
        x = torch.cat([state, action, emotion], dim=1)
        q = self.net(x)

        # Clamp Q-values to avoid extreme outputs
        return torch.clamp(q, -1e6, 1e6)


class PrioritizedReplayBuffer:
    """
    A prioritized experience replay buffer for reinforcement learning.

    Experiences with higher temporal-difference (TD) error are given higher probability to be sampled,
    improving learning efficiency by focusing on more informative transitions.

    Args:
        capacity (int): Maximum number of experiences the buffer can hold.
        alpha (float): Priority exponent. Higher values increase prioritization impact (0 = uniform).
        epsilon (float): Small constant to avoid zero priority and division errors.
    """

    def __init__(self, capacity=100000, alpha=0.6, epsilon=1e-4):
        self.capacity = capacity                      # Maximum buffer size
        self.alpha = alpha                            # Priority scaling factor
        self.epsilon = epsilon                        # Small constant for numerical stability
        self.buffer = []                              # Experience buffer
        self.priorities = []                          # List of priority values for sampling

    def add(self, transition, priority):
        """
        Add a transition with a given priority to the replay buffer.

        Args:
            transition (tuple): A single experience (state, action, reward, next_state, done).
            priority (float): TD error or other importance measure for prioritization.
        """
        # Compute scaled priority using alpha and epsilon for stability
        p = (abs(priority) + self.epsilon) ** self.alpha
        p = np.nan_to_num(p, nan=self.epsilon)  # ✅ Replace NaNs with small constant

        if len(self.buffer) < self.capacity:
            # Append new experience if buffer not full
            self.buffer.append(transition)
            self.priorities.append(p)
        else:
            # Replace the experience with the lowest priority
            idx = np.argmin(self.priorities)
            self.buffer[idx] = transition
            self.priorities[idx] = p

    def sample(self, batch_size):
        """
        Sample a batch of experiences based on priority probabilities.

        Args:
            batch_size (int): Number of experiences to sample.

        Returns:
            list: A list of sampled experiences.
        """
        priorities = np.array(self.priorities, dtype=np.float32)
        total = priorities.sum()

        # ✅ Ensure priorities sum is valid and not NaN or zero
        if total <= 0 or np.isnan(total):
            priorities += self.epsilon
            total = priorities.sum()

        # Normalize priorities into a probability distribution
        probs = priorities / total
        probs = np.nan_to_num(probs, nan=1.0 / len(probs))  # ✅ Replace NaNs with uniform probability

        # Sample indices based on the probability distribution
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)

        # Return the corresponding experiences
        return [self.buffer[i] for i in indices]

    def __len__(self):
        """
        Returns:
            int: Current number of stored experiences.
        """
        return len(self.buffer)


class DDPGAgent:
    """
    Deep Deterministic Policy Gradient (DDPG) agent with emotion modulation and prioritized experience replay.

    Attributes:
        env (GymEnv): The environment the agent interacts with.
        device (str): Device to run computations on (CPU or GPU).
        use_td_error (bool): Whether to use TD error for prioritized experience replay.
        actor, critic: Main actor and critic networks.
        actor_target, critic_target: Target networks used for soft updates.
        memory: Experience replay buffer (prioritized if use_td_error is True).
        emotion: Emotion module that returns current emotion state.
        ou_noise: Ornstein-Uhlenbeck process used for exploration noise.
    """
    def __init__(self, env, device='cuda' if torch.cuda.is_available() else 'cpu', use_td_error=False):
        """
        Initializes the DDPG agent, models, optimizers, and buffer.

        Args:
            env: Environment object with state_dim, action_dim, and bounds attributes.
            device (str): Computation device (CPU/GPU).
            use_td_error (bool): Whether to use TD error in prioritized replay.
        """
        param_bounds = env.bounds
        self.env = env
        self.device = device
        self.use_td_error = use_td_error

        self.emotion = EmotionModuleNone()

        # Actor networks
        self.actor = Actor(env.state_dim, env.action_dim, param_bounds).to(device)
        self.actor_target = Actor(env.state_dim, env.action_dim, param_bounds).to(device)
        self.actor_target.load_state_dict(self.actor.state_dict())

        # Critic networks
        self.critic = Critic(env.state_dim, env.action_dim).to(device)
        self.critic_target = Critic(env.state_dim, env.action_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Hyperparameters
        self.gamma = 0.99
        self.tau = 0.01
        self.batch_size = 128
        self.memory_size = 100000
        self.actor_lr = 1e-4
        self.critic_lr = 1e-3

        self.train_step = 1
        self.lr_decay_rate = 0.999

        self.target_update_freq = 1
        self.learn_step = 0
        self.actor_update_freq = 1

        # Optimizers
        self.actor_optim = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optim = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        # Replay memory and exploration noise
        self.memory = PrioritizedReplayBuffer(capacity=self.memory_size)
        self.ou_noise = OUNoise(env.action_dim)

    def act(self, state, add_noise=True):
        """
        Selects an action given the current state and optionally adds exploration noise.

        Args:
            state (np.ndarray): Current state of the environment.
            add_noise (bool): Whether to add OU exploration noise.

        Returns:
            np.ndarray: Action to execute.
        """
        emotion_state = self.emotion.get_emotion()
        state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        emotion_tensor = torch.from_numpy(emotion_state).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            scaled_action = self.actor(state_tensor, emotion_tensor)

        if not add_noise:
            return scaled_action.cpu().numpy().flatten()

        noise = self.ou_noise.sample()
        scale = self.actor.scale_params.cpu().numpy()
        offset = self.actor.offset_params.cpu().numpy()

        final_action = np.clip(
            scaled_action.cpu().numpy().flatten() + noise * scale,
            offset - scale,
            offset + scale
        )
        return final_action

    def remember(self, state, action, reward, next_state, done):
        """
        Stores a transition in memory, with optional TD error for prioritization.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Executed action.
            reward (float): Reward received.
            next_state (np.ndarray): Next state observed.
            done (bool): Whether the episode ended.
        """
        emotion_state = self.emotion.get_emotion()

        # Normalize action to [-1, 1]
        action = np.array(action)
        norm_action = (action - self.actor.offset_params.cpu().numpy()) / self.actor.scale_params.cpu().numpy()
        norm_action = np.clip(norm_action, -1.0, 1.0)

        td_error = 0
        if self.use_td_error:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
                action_tensor = torch.FloatTensor(norm_action).unsqueeze(0).to(self.device)
                emotion_tensor = torch.FloatTensor(emotion_state).unsqueeze(0).to(self.device)

                next_emotion_tensor = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)
                next_action = self.actor_target(next_state_tensor, next_emotion_tensor)
                target_q = self.critic_target(next_state_tensor, next_action, next_emotion_tensor)
                expected_q = reward + (1 - int(done)) * self.gamma * target_q.item()
                current_q = self.critic(state_tensor, action_tensor, emotion_tensor).item()
                td_error = abs(expected_q - current_q)

        self.memory.add(
            transition=(state, norm_action, reward, next_state, done, emotion_state, td_error),
            priority=td_error
        )

    def compute_dynamic_lr(emotion, base_lr, role="critic", min_lr=5e-5, max_lr=2e-3):
        """
        Computes a learning rate dynamically based on emotion input.

        Args:
            emotion (list): [curiosity, conservativeness, anxiety].
            base_lr (float): Base learning rate.
            role (str): 'actor' or 'critic'.
            min_lr (float): Minimum learning rate.
            max_lr (float): Maximum learning rate.

        Returns:
            float: Adjusted learning rate.
        """
        anxiety, conservativeness, curiosity = emotion

        if role == "critic":
            factor = 1.0 - 0.5 * anxiety + 0.4 * conservativeness + 0.1 * curiosity
        elif role == "actor":
            factor = 1.0 + 0.4 * curiosity - 0.3 * conservativeness
        else:
            factor = 1.0

        return float(np.clip(base_lr * factor, min_lr, max_lr))

    def learn(self):
        """
        Updates actor and critic networks using sampled transitions from replay buffer.

        Returns:
            dict: Loss values for actor and critic networks.
        """
        if len(self.memory) < self.batch_size:
            return

        # Sample batch
        batch = self.memory.sample(self.batch_size) if self.use_td_error else random.sample(self.memory.buffer, self.batch_size)
        states, actions, rewards, next_states, dones, emotions, _ = zip(*batch)

        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.FloatTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).unsqueeze(1).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).unsqueeze(1).to(self.device)
        emotions = torch.FloatTensor(emotions).to(self.device)

        # Reward scaling
        rewards *= 0.01

        # Compute emotion factors
        anxiety = torch.mean(emotions[:, 2]).item()
        conservativeness = torch.mean(emotions[:, 1]).item()
        curiosity = torch.mean(emotions[:, 0]).item()

        actor_factor = 1.0 + 0.8 * curiosity - 0.4 * conservativeness + 0.2 * anxiety
        critic_factor = 1.0 - 0.3 * anxiety + 0.6 * conservativeness

        decay_factor = self.lr_decay_rate ** self.train_step
        critic_scaled_lr = float(np.clip(self.critic_lr * critic_factor * decay_factor, self.critic_lr * 0.1, self.critic_lr * 3))
        actor_scaled_lr = float(np.clip(self.actor_lr * actor_factor * decay_factor, self.actor_lr * 0.1, self.actor_lr * 3))

        for g in self.critic_optim.param_groups:
            g["lr"] = critic_scaled_lr
        for g in self.actor_optim.param_groups:
            g["lr"] = actor_scaled_lr

        # Critic update
        with torch.no_grad():
            next_emotion_tensor = torch.FloatTensor(self.emotion.get_emotion()).unsqueeze(0).to(self.device)
            next_emotion_tensor = next_emotion_tensor.expand(next_states.size(0), -1)
            next_actions = self.actor_target(next_states, next_emotion_tensor)
            target_q = self.critic_target(next_states, next_actions, next_emotion_tensor)
            target_q = rewards + (1 - dones) * self.gamma * target_q

        current_q = self.critic(states, actions, emotions)
        critic_loss = nn.SmoothL1Loss()(current_q, target_q)

        self.critic_optim.zero_grad()
        critic_loss.backward()
        nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=5.0)
        self.critic_optim.step()

        actor_loss = None
        if self.learn_step % self.actor_update_freq == 0:
            action_out = self.actor(states, emotions)
            actor_loss = -self.critic(states, action_out, emotions).mean()

            # Diversity bonus
            diversity_term = torch.std(action_out, dim=1).mean()
            actor_loss -= 0.03 * diversity_term

            # L2 regularization
            l2_lambda = 1e-4
            l2_reg = torch.tensor(0., device=self.device)
            for param in self.actor.parameters():
                l2_reg += torch.norm(param, p=2)
            actor_loss += l2_lambda * l2_reg

            self.actor_optim.zero_grad()
            actor_loss.backward()
            nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=5.0)
            self.actor_optim.step()

        # Soft update target networks
        self.learn_step += 1
        if self.learn_step % self.target_update_freq == 0:
            for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.critic_target.parameters(), self.critic.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            "critic_loss": critic_loss.item() if critic_loss is not None else None,
            "actor_loss": actor_loss.item() if actor_loss is not None else None
        }

    def load_weights(self, weights):
        """Loads weights into actor and critic networks."""
        self.actor.load_state_dict(weights['actor'])
        self.critic.load_state_dict(weights['critic'])

    def get_weights(self):
        """Returns current actor and critic weights."""
        return {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict()
        }

    def save(self, path):
        """Saves model and optimizer states to a file."""
        save_dict = {
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_target': self.actor_target.state_dict(),
            'critic_target': self.critic_target.state_dict(),
            'actor_optim': self.actor_optim.state_dict(),
            'critic_optim': self.critic_optim.state_dict(),
            'train_step': self.train_step,
        }

        if hasattr(self, "emotion") and hasattr(self.emotion, "transformer"):
            save_dict['emotion_transformer'] = self.emotion.transformer.state_dict()

        torch.save(save_dict, path)

    def load(self, filename):
        """Loads model and optimizer states from a file."""
        checkpoint = torch.load(filename, map_location=self.device)

        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])

        if 'emotion' in checkpoint:
            self.emotion.current_emotion = checkpoint['emotion']
        else:
            print("⚠️ No emotion state found in checkpoint.")

        if 'emotion_transformer' in checkpoint:
            if hasattr(self.emotion, 'transformer'):
                self.emotion.transformer.load_state_dict(checkpoint['emotion_transformer'])
                print("✅ EmotionTransformer loaded.")
            else:
                print("⚠️ EmotionTransformer in checkpoint not used in current config.")


def reset_actor_scaling(agent, new_bounds):
    """
    Reset the scaling parameters (scale_params and offset_params) of both the actor
    and target actor networks based on the updated action bounds.

    Args:
        agent: DDPGAgent instance containing actor and actor_target networks.
        new_bounds (dict): New action bounds, typically from env.bounds.
    """
    if hasattr(agent.actor, '_init_param_scaling'):
        agent.actor._init_param_scaling(new_bounds)
    if hasattr(agent.actor_target, '_init_param_scaling'):
        agent.actor_target._init_param_scaling(new_bounds)
    print("✅ Actor scaling parameters re-initialized based on updated bounds.")


def train_ddpg(env, agent, episodes=1000, max_steps=500, log_prefix="ddpg_emotion", logger=None):
    """
    Train a DDPG agent with emotion-driven modulation in a given environment.

    Args:
        env: The environment instance implementing reset() and step().
        agent: The DDPGAgent instance (with emotion and prioritized replay buffer).
        episodes (int): Total number of training episodes.
        max_steps (int): Maximum number of steps per episode.
        log_prefix (str): Prefix name for logging and TensorBoard tracking.
        logger (logging.Logger): Optional logger instance.

    Returns:
        List of episode rewards.
    """
    if logger is None:
        logger = init_logger(log_prefix=log_prefix)

    rewards = []
    env.max_steps = max_steps  # Set environment's internal step limit

    # Set up TensorBoard directory
    tb_log_dir = f"runs/{log_prefix}"
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)  # Remove existing logs
    writer = SummaryWriter(log_dir=tb_log_dir)

    global_step = 0
    pretrain_path = "pre_train_pth/AIM_v1_sim_24900_25000.pth"
    use_pretrain = 0

    # === Pretrained model loading logic ===
    if not use_pretrain:
        logger.info("Pretraining disabled. Training from scratch.")
        print("Pretraining disabled. Training from scratch.")
    elif os.path.exists(pretrain_path):
        weights = torch.load(pretrain_path, weights_only=False)
        agent.load_weights(weights)
        reset_actor_scaling(agent, env.bounds)
        logger.info(f"Pretrained weights loaded from: {pretrain_path}")
        print(f"Pretrained weights loaded from: {pretrain_path}")
    else:
        logger.info("Pretrained weights not found. Training from scratch.")
        print("Pretrained weights not found. Training from scratch.")

    # Start timer for benchmarking every 100 episodes
    start_100 = time.time()

    # === Main training loop ===
    for ep in tqdm(range(episodes), desc="Training Progress", ncols=100):
        state = env.reset()
        episode_reward = 0
        average_step = 0

        agent.ou_noise.reset()  # Reset exploration noise
        agent.ou_noise.anneal()  # Optional noise annealing

        for step in range(max_steps):
            # === Agent action and environment interaction ===
            action = np.transpose(agent.act(state))
            next_state, reward, done, info = env.step(action)

            movement = next_state - state
            agent.emotion.update(reward, movement)  # Update emotion module

            agent.remember(state, action, reward, next_state, done)  # Store transition
            state = next_state
            episode_reward += reward

            # === Learning update ===
            loss = agent.learn()
            if loss:
                if loss.get('actor_loss') is not None:
                    writer.add_scalar("Loss/Actor", loss['actor_loss'], global_step)
                if loss.get('critic_loss') is not None:
                    writer.add_scalar("Loss/Critic", loss['critic_loss'], global_step)
                writer.add_scalar("LR/Actor", agent.actor_optim.param_groups[0]['lr'], global_step)
                writer.add_scalar("LR/Critic", agent.critic_optim.param_groups[0]['lr'], global_step)

            # === Log emotion dynamics ===
            cur_emotion = agent.emotion.get_emotion()
            if isinstance(cur_emotion, torch.Tensor):
                cur_emotion = cur_emotion.detach().cpu().numpy().squeeze()
            cur_emotion = np.array(cur_emotion).flatten()
            writer.add_scalar("Emotion/Curiosity", float(cur_emotion[0]), global_step)
            writer.add_scalar("Emotion/Conservativeness", float(cur_emotion[1]), global_step)
            writer.add_scalar("Emotion/Anxiety", float(cur_emotion[2]), global_step)

            # Periodic logging per episode
            if step % (max_steps // 10) == 0 or step == max_steps - 1:
                logger.info(
                    f"Ep {ep:03d} Step {step:03d} | Reward: {reward:.2f} | Loss: {loss} | Emotion: {np.round(cur_emotion, 2)}"
                )

            global_step += 1
            average_step += 1

            if done:
                print(f"✔️ Episode {ep} finished early at step {step}")
                break

        # === Logging episode-level metrics ===
        rewards.append(episode_reward)
        writer.add_scalar("Reward/Episode", episode_reward, ep)
        writer.add_scalar("Reward/Episode(average)", episode_reward / average_step, ep)
        writer.add_scalar("Metric/Weight", info["final_weight"], ep)
        writer.add_scalar("Metric/Error", info["weight_error"], ep)
        writer.add_scalar("Metric/Time", info["total_time"], ep)
        writer.add_scalar("Reward/Step", reward, ep)

        # Log action components
        for key, value in info["action"].items():
            writer.add_scalar(f"Action/{key}", value, ep)
        writer.add_scalar("Metric/SlowWeight", info["action"].get("slow_weight", 0.0), ep)

        # === Episode summary log ===
        if ep % 1 == 0 or ep == episodes - 1:
            cur_emotion = agent.emotion.get_emotion()
            if isinstance(cur_emotion, torch.Tensor):
                cur_emotion = cur_emotion.detach().cpu().numpy().squeeze()
            cur_emotion = np.array(cur_emotion).flatten()

            actor_loss = f"{loss.get('actor_loss'):.4f}" if loss and loss.get("actor_loss") is not None else "-"
            critic_loss = f"{loss.get('critic_loss'):.4f}" if loss and loss.get("critic_loss") is not None else "-"
            slow_weight = info["action"].get("slow_weight", 0.0)

            logger.info(
                f"[Episode {ep:04d}] "
                f"TotalReward: {episode_reward:.2f} | "
                f"AvgReward: {episode_reward / average_step:.2f} | "
                f"WeightErr: {info['weight_error']:.2f}g | "
                f"Time: {info['total_time']:.2f}s | "
                f"SlowWeight: {slow_weight:.2f}g | "
                f"Emotion: [Curi: {cur_emotion[0]:.2f}, Cons: {cur_emotion[1]:.2f}, Anx: {cur_emotion[2]:.2f}] | "
                f"ActorLoss: {actor_loss} | CriticLoss: {critic_loss}"
            )

        # Log every 100 episodes: runtime benchmark
        if ep > 0 and ep % 100 == 0:
            duration_100 = time.time() - start_100
            logger.info(
                f"⏱️ Episodes {ep - 100} to {ep} took {duration_100:.2f} seconds, average {duration_100 / 100:.2f} sec/episode")
            start_100 = time.time()

        # Save model weights every 10 episodes
        # if ep % 10 == 0 or ep == episodes - 1:
            # model_dir = os.path.join("saved_models", log_prefix)
            # torch.save(agent.get_weights(),"saved_models/"+log_prefix+".pth")
            # logger.info(f"[Episode {ep}] Model weights saved to ddpg_emotion_agent.pth")

        agent.train_step += 1

    writer.close()
    return rewards


class OUNoise:
    """
    Ornstein-Uhlenbeck process for generating temporally correlated noise.

    This is commonly used in continuous action space environments (e.g., DDPG)
    to add exploration noise that is more consistent between time steps,
    mimicking Brownian motion with momentum.

    Args:
        size (int): Dimensionality of the noise vector (same as action size).
        mu (float): Long-term mean (default: 0.0).
        theta (float): Rate of mean reversion (default: 0.15).
        sigma (float): Initial volatility or scale of noise (default: 0.6).
        min_sigma (float): Minimum allowed sigma for annealing (default: 0.1).
        decay (float): Multiplicative decay factor for sigma after each episode (default: 0.9995).
    """

    def __init__(self, size, mu=0.0, theta=0.15, sigma=0.6, min_sigma=0.1, decay=0.9995):
        self.size = size
        self.mu = mu * np.ones(size, dtype=np.float32)  # Target mean vector
        self.theta = theta  # Mean reversion speed
        self.sigma = sigma  # Initial noise scale
        self.min_sigma = min_sigma  # Lower bound on sigma
        self.decay = decay  # Sigma decay rate per episode
        self.reset()  # Initialize internal noise state

    def reset(self):
        """
        Reset the internal noise state to the mean value.
        Should be called at the start of each episode.
        """
        self.state = np.copy(self.mu)

    def sample(self):
        """
        Generate a noise sample based on the current internal state.

        Returns:
            np.ndarray: A noise vector to be added to the action.
        """
        dx = self.theta * (self.mu - self.state)  # Pull toward the mean
        dx += self.sigma * np.random.randn(self.size).astype(np.float32)  # Add random fluctuation
        self.state += dx
        return self.state

    def anneal(self):
        """
        Gradually reduce the volatility (sigma) after each episode
        to shift from exploration to exploitation.
        """
        self.sigma = max(self.min_sigma, self.sigma * self.decay)
