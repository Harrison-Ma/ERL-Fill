import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class RolloutBuffer:
    """
    RolloutBuffer stores the trajectories (rollouts) collected from environment interactions
    for use in policy gradient algorithms such as PPO.

    Attributes:
        states (List): Collected environment states.
        actions (List): Actions taken in response to the states.
        log_probs (List): Log probabilities of the actions under the policy.
        rewards (List): Rewards received after each action.
        dones (List): Boolean flags indicating episode termination.
        values (List): Estimated state values from the critic.
        constraints (List): Optional constraint values (e.g., costs or risks) for constrained RL.
    """

    def __init__(self):
        """Initializes empty buffers for storing rollout data."""
        self.states = []       # List of observed states
        self.actions = []      # List of actions taken
        self.log_probs = []    # Log probabilities of actions
        self.rewards = []      # Received rewards
        self.dones = []        # Episode done flags
        self.values = []       # Estimated values from the critic
        self.constraints = []  # Estimated constraint values (if used)

    def clear(self):
        """
        Clears all stored rollout data.

        This method reinitializes all buffer lists and is typically called
        at the end of a training epoch or episode.
        """
        self.__init__()


class ActorCritic(nn.Module):
    """
    Actor-Critic neural network module for reinforcement learning.

    This network includes:
        - An actor network that outputs the mean and standard deviation of actions.
        - A critic network that estimates the state value.
        - An optional constraint critic for constrained reinforcement learning.

    Args:
        state_dim (int): Dimension of the input state.
        action_dim (int): Dimension of the action output.
        hidden_dim (int): Number of hidden units in each layer (default: 256).
    """

    def __init__(self, state_dim, action_dim, hidden_dim=256):
        super().__init__()

        # Actor network to predict the mean of the action distribution
        self.actor_mean = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

        # Actor network to predict the log standard deviation of the action distribution
        self.actor_log_std = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, action_dim), nn.Tanh()  # Tanh bounds log_std
        )

        # Critic network to estimate the value of the current state
        self.critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # Constraint critic estimates the cost or constraint value (used in constrained RL)
        self.constraint_critic = nn.Sequential(
            nn.Linear(state_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def get_action_distribution(self, state):
        """
        Constructs a Gaussian action distribution given the input state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            torch.distributions.Normal: A Normal distribution over actions.
        """
        mean = self.actor_mean(state)
        log_std = self.actor_log_std(state).clamp(-20, 2)  # Clamp for numerical stability
        std = torch.exp(log_std)  # Convert log_std to standard deviation
        return torch.distributions.Normal(mean, std)

    def act(self, state, deterministic=False):
        """
        Samples or returns a deterministic action from the policy.

        Args:
            state (torch.Tensor): The input state tensor.
            deterministic (bool): If True, returns the mean action. Otherwise, samples from the distribution.

        Returns:
            torch.Tensor: The action tensor, squashed with tanh to bound it in [-1, 1].
        """
        dist = self.get_action_distribution(state)
        if deterministic:
            action = dist.mean  # Use the mean for deterministic action
        else:
            action = dist.rsample()  # Sample using reparameterization trick
        action = torch.tanh(action)  # Bound the action output
        return action

    def evaluate(self, state):
        """
        Evaluates both value functions (reward and constraint) for a given state.

        Args:
            state (torch.Tensor): The input state tensor.

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: Estimated value and constraint value.
        """
        value = self.critic(state)                # Value estimate for standard reward
        constraint_value = self.constraint_critic(state)  # Value estimate for constraint/cost
        return value, constraint_value


class PPOLagrangian:
    """
    Proximal Policy Optimization (PPO) agent with Lagrangian-based constraint optimization.

    This implementation extends standard PPO to handle constrained reinforcement learning tasks,
    using a Lagrange multiplier to balance constraint satisfaction with reward maximization.

    Args:
        env: The environment object providing state/action space and bounds.
        device (str): PyTorch device to use ('cuda' or 'cpu').
    """

    def __init__(self, env, device='cuda'):
        self.env = env
        self.device = device
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        # PPO hyperparameters
        self.gamma = 0.99                      # Discount factor for future rewards
        self.lam = 0.95                        # GAE (Generalized Advantage Estimation) lambda
        self.eps_clip = 0.2                    # Clipping parameter for PPO
        self.lr = 3e-4                         # Learning rate
        self.update_epochs = 10               # Number of epochs per PPO update
        self.batch_size = 64                  # Batch size for updates

        # Constraint-specific settings
        self.constraint_limit = 0.01          # Allowed threshold for constraint violation
        self.lambda_lr = 0.05                 # Lagrange multiplier update rate

        # Actor-Critic policy network and optimizer
        self.policy = ActorCritic(self.state_dim, self.action_dim).to(device)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=self.lr)

        # Buffer to store rollout trajectories
        self.rollout_buffer = RolloutBuffer()

        # Lagrange multiplier for constraint handling
        self.lagrange_multiplier = torch.tensor(0.0, requires_grad=False, device=device)

        # Memory interface for environment compatibility
        self.memory = self

    def __len__(self):
        """
        Returns the number of stored experiences in the rollout buffer.
        Enables compatibility with memory length checks.
        """
        return len(self.rollout_buffer.states)

    def normalize_action(self, action):
        """
        Normalizes actions to [-1, 1] based on environment bounds.

        Args:
            action (np.ndarray): Original action in real-world scale.

        Returns:
            np.ndarray: Normalized action.
        """
        return np.array([
            2 * (action[i] - low) / (high - low) - 1
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def denormalize_action(self, norm_action):
        """
        Converts normalized actions back to original scale.

        Args:
            norm_action (np.ndarray): Normalized action in [-1, 1].

        Returns:
            np.ndarray: Real-world scaled action.
        """
        return np.array([
            0.5 * (norm_action[i] + 1) * (high - low) + low
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def act(self, state, deterministic=False):
        """
        Selects an action given the current state.

        Args:
            state (np.ndarray): Current environment state.
            deterministic (bool): Whether to use deterministic actions.

        Returns:
            np.ndarray: Selected action in environment's original scale.
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.policy.act(state_tensor, deterministic)
        return self.denormalize_action(action.detach().cpu().numpy().flatten())

    def remember(self, state, action, reward, next_state, done, cost):
        """
        Stores a single transition in the rollout buffer.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Reward received.
            next_state (np.ndarray): Next state after action.
            done (bool): Whether episode terminated.
            cost (float): Cost (constraint violation signal).
        """
        norm_action = self.normalize_action(action)
        self.rollout_buffer.states.append(torch.FloatTensor(state))
        self.rollout_buffer.actions.append(torch.FloatTensor(norm_action))
        self.rollout_buffer.rewards.append(torch.FloatTensor([reward]))
        self.rollout_buffer.dones.append(torch.FloatTensor([done]))

        value, constraint = self.policy.evaluate(torch.FloatTensor(state).unsqueeze(0).to(self.device))
        self.rollout_buffer.values.append(value.detach().cpu().squeeze(0))
        self.rollout_buffer.constraints.append(torch.FloatTensor([cost]))

    def update(self):
        """
        Performs a PPO policy update using the stored rollout buffer.

        Returns:
            dict: Training metrics including losses and entropy.
        """
        # Convert buffer data to tensors
        states = torch.stack(self.rollout_buffer.states).to(self.device)
        actions = torch.stack(self.rollout_buffer.actions).to(self.device)
        rewards = torch.stack(self.rollout_buffer.rewards).to(self.device)
        dones = torch.stack(self.rollout_buffer.dones).to(self.device)
        values = torch.stack(self.rollout_buffer.values).to(self.device)
        constraints = torch.stack(self.rollout_buffer.constraints).to(self.device)

        # Generalized Advantage Estimation (GAE) for rewards and constraints
        returns = []
        advantages = []
        constraint_returns = []
        gae = 0
        constraint_gae = 0
        next_value = 0
        next_constraint = 0

        for i in reversed(range(len(rewards))):
            # Reward advantage
            delta = rewards[i] + self.gamma * next_value * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lam * (1 - dones[i]) * gae
            advantages.insert(0, gae + values[i])
            next_value = values[i]

            # Constraint advantage
            constraint_delta = constraints[i] + self.gamma * next_constraint * (1 - dones[i]) - constraints[i]
            constraint_gae = constraint_delta + self.gamma * self.lam * (1 - dones[i]) * constraint_gae
            constraint_returns.insert(0, constraint_gae + constraints[i])
            next_constraint = constraints[i]

        advantages = torch.stack(advantages)
        constraint_returns = torch.stack(constraint_returns)

        # PPO update loop
        for _ in range(self.update_epochs):
            indices = np.random.permutation(len(rewards))
            for start in range(0, len(rewards), self.batch_size):
                end = start + self.batch_size
                batch_indices = indices[start:end]

                batch_states = states[batch_indices]
                batch_actions = actions[batch_indices]
                batch_adv = advantages[batch_indices]
                batch_const = constraint_returns[batch_indices]

                curr_values, curr_constraints = self.policy.evaluate(batch_states)
                dist = self.policy.get_action_distribution(batch_states)
                log_probs = dist.log_prob(batch_actions).sum(axis=-1)

                # PPO clipped surrogate objective
                ratio = torch.exp(log_probs - log_probs.detach())
                surrogate1 = ratio * batch_adv
                surrogate2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * batch_adv
                loss_clip = -torch.min(surrogate1, surrogate2).mean()

                value_loss = nn.MSELoss()(curr_values.squeeze(), batch_adv.squeeze())
                constraint_loss = nn.MSELoss()(curr_constraints.squeeze(), batch_const.squeeze())

                # Lagrangian penalty for constraint
                lagrangian_term = self.lagrange_multiplier * constraint_loss
                total_loss = loss_clip + 0.5 * value_loss + lagrangian_term

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

        # Update Lagrange multiplier based on constraint satisfaction
        mean_constraint = torch.mean(constraint_returns).item()
        self.lagrange_multiplier += self.lambda_lr * (mean_constraint - self.constraint_limit)
        self.lagrange_multiplier = torch.clamp(self.lagrange_multiplier, 0.0, 10.0)

        self.rollout_buffer.clear()
        return {
            'policy_loss': loss_clip.item(),
            'value_loss': value_loss.item(),
            'constraint_loss': constraint_loss.item(),
            'lambda': self.lagrange_multiplier.item(),
            'entropy': dist.entropy().mean().item()
        }

    def save(self, path):
        """
        Saves the model parameters and Lagrange multiplier to a file.

        Args:
            path (str): Path to save the model checkpoint.
        """
        torch.save({
            'policy_state_dict': self.policy.state_dict(),
            'lagrange_multiplier': self.lagrange_multiplier
        }, path)

    def load(self, path):
        """
        Loads the model parameters and Lagrange multiplier from a checkpoint file.

        Args:
            path (str): Path to the checkpoint file.
        """
        ckpt = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ckpt['policy_state_dict'])
        self.lagrange_multiplier = ckpt.get('lagrange_multiplier', torch.tensor(0.0)).to(self.device)


def train_ppo_lagrangian(env, agent, episodes=1000, max_steps=500, log_prefix="ppo_lagrangian_exp", model_path=None, pretrain_path=None):
    """
    Trains a PPO-Lagrangian agent on the given environment.

    This function handles the training loop, logging (to both file and TensorBoard),
    model checkpointing, and optional loading of pretrained weights.

    Args:
        env: The environment object following OpenAI Gym-style interface.
        agent: An instance of the PPOLagrangian class.
        episodes (int): Total number of training episodes.
        max_steps (int): Maximum steps per episode.
        log_prefix (str): Prefix for logging and checkpoint file names.
        model_path (str or None): Optional custom path to save intermediate models.
        pretrain_path (str or None): Optional path to load pretrained model.

    Returns:
        List[float]: A list containing total rewards for each episode.
    """
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # Setup directories and paths
    tb_log_dir = f"runs/{log_prefix}"  # TensorBoard log directory
    log_file_path = f"logs/ppo_lagrangian_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.pth")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.pth")

    # Logging setup
    logger = logging.getLogger(f"ppo_lagrangian_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Clear previous TensorBoard logs if exist
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Load pretrained model if provided
    if pretrain_path and os.path.exists(pretrain_path):
        agent.load(pretrain_path)
        print(f"✅ Successfully loaded pretrained model: {pretrain_path}")
        logger.info(f"Loaded pretrained model: {pretrain_path}")
    else:
        print("❌ No pretrained model loaded. Training from scratch.")
        logger.info("Training from scratch without pretrained model.")

    # Main training loop
    for ep in tqdm(range(episodes), desc="PPO-Lagrangian Training", ncols=100):
        state = env.reset()
        ep_reward = 0
        average_step = 0

        for step in range(max_steps):
            action = agent.act(state, deterministic=False)
            next_state, reward, done, info = env.step(action)

            # Constraint signal (e.g. energy consumption, safety metric, etc.)
            cost = info.get("constraint_cost", 0.0)
            agent.remember(state, action, reward, next_state, done, cost)

            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # Log step-level metrics to TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Cost/Step", cost, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)

            if done:
                break

        # Perform PPO update after episode
        update_info = agent.update()

        rewards.append(ep_reward)
        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", ep_reward / average_step, ep)

        logger.info(
            f"[Episode {ep:04d}] TotalReward: {ep_reward:.2f} | AvgReward: {ep_reward / average_step:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s"
        )

        # Log training info (losses, entropy, lambda)
        if update_info:
            logger.info(
                f"λ: {update_info.get('lambda', '-')} | PolicyLoss: {update_info.get('policy_loss', '-'):.2f} | "
                f"ValueLoss: {update_info.get('value_loss', '-'):.2f} | Entropy: {update_info.get('entropy', '-'):.2f}"
            )
            writer.add_scalar("Loss/Policy", update_info.get('policy_loss', 0.0), ep)
            writer.add_scalar("Loss/Value", update_info.get('value_loss', 0.0), ep)
            writer.add_scalar("Loss/Entropy", update_info.get('entropy', 0.0), ep)
            writer.add_scalar("Lambda", update_info.get('lambda', 0.0), ep)

        # Save checkpoint periodically and on final episode
        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.pth")
            agent.save(ckpt_path)
            logger.info(f"[Episode {ep:04d}] Model checkpoint saved at {ckpt_path}")

    # Save final model after training ends
    agent.save(final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")
    writer.close()

    return rewards

