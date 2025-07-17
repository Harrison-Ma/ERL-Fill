import numpy as np
import os
import shutil
import logging
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


class DummyBuffer:
    """
    A simple buffer class for storing arbitrary experience tuples.
    """
    def __init__(self):
        self.buffer = []

    def append(self, item):
        """
        Append a new item to the buffer.

        Args:
            item (Any): The item to store.
        """
        self.buffer.append(item)

    def __len__(self):
        """
        Return the current number of items in the buffer.
        """
        return len(self.buffer)

    def __getitem__(self, idx):
        """
        Get the item at a specific index.

        Args:
            idx (int): Index of the item to retrieve.
        """
        return self.buffer[idx]


class RLS_PIDAgent:
    """
    An agent that combines Recursive Least Squares (RLS) system identification
    with a PID (Proportional-Integral-Derivative) control strategy.
    """

    def __init__(self, env, device=None, lambda_=0.99, delta=1e5):
        """
        Initialize the RLS-PID agent.

        Args:
            env: The environment instance, expected to have `state_dim`, `action_dim`, and `bounds` attributes.
            device: Unused placeholder for potential device specification (e.g. torch/cuda).
            lambda_ (float): Forgetting factor for RLS update (close to 1.0).
            delta (float): Initial value for the RLS covariance matrix diagonal (controls initial uncertainty).
        """
        self.env = env
        self.state_dim = env.state_dim
        self.action_dim = env.action_dim
        self.bounds = env.bounds

        # RLS setup for identifying dynamics: [state, action, bias] → next_state[0]
        self.rls_dim = self.state_dim + self.action_dim + 1
        self.theta_rls = np.zeros((self.rls_dim, 1))  # Parameter vector
        self.P_rls = np.eye(self.rls_dim) * delta     # Covariance matrix
        self.lambda_rls = lambda_

        # PID control parameters
        self.Kp = 1.0
        self.Ki = 0.1
        self.Kd = 0.05

        # PID state
        self.integral = 0.0
        self.prev_error = 0.0

        self.memory = DummyBuffer()

    def normalize_action(self, action):
        """
        Normalize action values to the range [-1, 1] based on environment bounds.

        Args:
            action (np.ndarray): Raw action values.

        Returns:
            np.ndarray: Normalized actions.
        """
        return np.array([
            2 * (action[i] - low) / (high - low) - 1
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def denormalize_action(self, norm_action):
        """
        Convert normalized actions back to their actual range.

        Args:
            norm_action (np.ndarray): Normalized action values in [-1, 1].

        Returns:
            np.ndarray: Denormalized action values.
        """
        return np.array([
            0.5 * (norm_action[i] + 1) * (high - low) + low
            for i, (key, (low, high)) in enumerate(self.bounds.items())
        ])

    def reset(self):
        """
        Reset the PID controller's internal state (integral and previous error).
        """
        self.integral = 0.0
        self.prev_error = 0.0

    def act(self, state, deterministic=True):
        """
        Compute an action using PID control and optional safety filtering.

        Args:
            state (np.ndarray): Current environment state.
            deterministic (bool): Ignored (for compatibility).

        Returns:
            np.ndarray: Action vector.
        """
        # PID control logic
        target = getattr(self.env, 'target_weight', 25000)
        current_weight = state[0]
        error = target - current_weight

        self.integral += error
        derivative = error - self.prev_error
        output = self.Kp * error + self.Ki * self.integral + self.Kd * derivative
        self.prev_error = error

        # Generate and clip action
        action = np.clip(np.ones(self.action_dim) * output, -1, 1)
        action = self.denormalize_action(action)

        # Safety check: override action if predicted next state is risky
        predicted_next = self.predict_next_state(state, action)
        safe_threshold = getattr(self.env, 'error_tolerance', 50)
        if abs(target - predicted_next) > 2 * safe_threshold:
            action = self.safe_action(state)

        return action

    def predict_next_state(self, state, action):
        """
        Predict the next state's first dimension using current RLS model.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.

        Returns:
            float: Predicted next state's first component.
        """
        x_rls = np.hstack([state, action, 1.0]).reshape(-1, 1)
        pred = float(x_rls.T @ self.theta_rls)
        return pred

    def safe_action(self, state):
        """
        Return a predefined safe action (zero action vector).

        Args:
            state (np.ndarray): Current state.

        Returns:
            np.ndarray: Safe action (zeros).
        """
        return np.zeros(self.action_dim)

    def remember(self, state, action, reward, next_state, done, cost):
        """
        Store experience and update RLS model with the observed transition.

        Args:
            state (np.ndarray): Current state.
            action (np.ndarray): Action taken.
            reward (float): Received reward.
            next_state (np.ndarray): Next state observed.
            done (bool): Whether the episode has terminated.
            cost (float): Cost signal (e.g. constraint violation).
        """
        x_rls = np.hstack([state, action, 1.0]).reshape(-1, 1)
        y_rls = np.array([next_state[0]]).reshape(1, 1)

        # RLS parameter update
        Px = self.P_rls @ x_rls
        gain = Px / (self.lambda_rls + x_rls.T @ Px)
        self.theta_rls += gain @ (y_rls - x_rls.T @ self.theta_rls)
        self.P_rls = (self.P_rls - gain @ x_rls.T @ self.P_rls) / self.lambda_rls

        self.memory.append((state, action, reward, next_state, done, cost))

    def update(self):
        """
        Optionally perform updates after each episode (for logging or tuning).

        Returns:
            dict: Information about the current PID and RLS states.
        """
        return {
            'rls_theta_norm': np.linalg.norm(self.theta_rls),
            'integral_term': self.integral,
            'last_error': self.prev_error
        }

    def save(self, path):
        """
        Save the RLS and PID parameters to a file.

        Args:
            path (str): Destination path (saved in `.npz` format).
        """
        np.savez(path,
                 theta_rls=self.theta_rls,
                 P_rls=self.P_rls,
                 Kp=self.Kp,
                 Ki=self.Ki,
                 Kd=self.Kd)

    def load(self, path):
        """
        Load the RLS and PID parameters from a `.npz` file.

        Args:
            path (str): Path to the saved file.
        """
        data = np.load(path)
        self.theta_rls = data['theta_rls']
        self.P_rls = data['P_rls']
        self.Kp = float(data['Kp'])
        self.Ki = float(data['Ki'])
        self.Kd = float(data['Kd'])


def train_rls_pid(env, agent, episodes=1000, max_steps=500, log_prefix="rls_pid_exp", model_path=None, pretrain_path=None):
    """
    Train an RLS-PID agent in a given environment.

    Args:
        env: The simulation environment, which should support Gym-like interface with `reset()` and `step()`.
        agent: An instance of RLS_PIDAgent responsible for action generation and learning.
        episodes (int): Number of episodes to train the agent for.
        max_steps (int): Maximum steps allowed per episode.
        log_prefix (str): Prefix string used for logging directories and file naming.
        model_path (str): Optional path to save intermediate models.
        pretrain_path (str): Optional path to load a pre-trained model before training.

    Returns:
        rewards (list): Episode-wise total rewards collected during training.
    """
    rewards = []
    env.max_steps = max_steps
    global_step = 0

    # Setup logging and checkpointing directories
    tb_log_dir = f"runs/{log_prefix}"
    log_file_path = f"logs/rls_pid_training_{log_prefix}.log"
    model_dir = os.path.join("saved_models", log_prefix)
    os.makedirs(model_dir, exist_ok=True)

    final_model_path = os.path.join(model_dir, f"{log_prefix}_final.npz")
    if model_path is None:
        model_path = os.path.join(model_dir, f"{log_prefix}_ep0000.npz")

    # Configure file logger
    logger = logging.getLogger(f"rls_pid_logger_{log_prefix}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if not logger.handlers:
        fh = logging.FileHandler(log_file_path, mode='w', encoding='utf-8')
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    # Clean up TensorBoard logs if they already exist
    if os.path.exists(tb_log_dir):
        shutil.rmtree(tb_log_dir)
    writer = SummaryWriter(log_dir=tb_log_dir)

    # Optionally load a pre-trained model
    if pretrain_path and os.path.exists(pretrain_path):
        agent.load(pretrain_path)
        print(f"✅ Successfully loaded pretrained model: {pretrain_path}")
        logger.info(f"Successfully loaded pretrained model: {pretrain_path}")
    else:
        print("❌ No pretrained model found. Training from scratch.")
        logger.info("No pretrained model found. Training from scratch.")

    # Main training loop
    for ep in tqdm(range(episodes), desc="RLS+PID Training", ncols=100):
        state = env.reset()
        agent.reset()
        ep_reward = 0
        average_step = 0
        unsafe_action_count = 0

        for step in range(max_steps):
            action = agent.act(state)

            # Predict next state to determine if action is "unsafe"
            predicted_next = agent.predict_next_state(state, action)
            target = getattr(env, 'target_weight', 25000)
            safe_threshold = getattr(env, 'error_tolerance', 50)
            if abs(target - predicted_next) > 2 * safe_threshold:
                unsafe_action_count += 1

            # Execute action
            next_state, reward, done, info = env.step(action)
            cost = info.get("constraint_cost", 0.0)
            agent.remember(state, action, reward, next_state, done, cost)

            # Log and update state
            state = next_state
            ep_reward += reward
            average_step += 1
            global_step += 1

            # Write per-step metrics to TensorBoard
            writer.add_scalar("Reward/Step", reward, global_step)
            writer.add_scalar("Cost/Step", cost, global_step)
            writer.add_scalar("Metric/Error", info.get("weight_error", 0.0), global_step)
            writer.add_scalar("Metric/Time", info.get("total_time", 0.0), global_step)

            if done:
                break

        # Perform optional post-episode update and log relevant metrics
        update_info = agent.update()
        avg_ep_reward = ep_reward / average_step if average_step > 0 else 0
        rewards.append(ep_reward)

        writer.add_scalar("Reward/Episode", ep_reward, ep)
        writer.add_scalar("Reward/Episode(average)", avg_ep_reward, ep)
        writer.add_scalar("Safety/UnsafeActions", unsafe_action_count, ep)

        if update_info:
            writer.add_scalar("Theta/RLS_Norm", update_info.get('rls_theta_norm', 0.0), ep)
            writer.add_scalar("PID/Integral", update_info.get('integral_term', 0.0), ep)
            writer.add_scalar("PID/LastError", update_info.get('last_error', 0.0), ep)

        # Write episode summary to log
        logger.info(
            f"[Ep {ep:04d}] Reward: {ep_reward:.2f} | Avg: {avg_ep_reward:.2f} | "
            f"WeightErr: {info.get('weight_error', 0.0):.2f}g | Time: {info.get('total_time', 0.0):.2f}s | "
            f"UnsafeActions: {unsafe_action_count}"
        )

        if update_info:
            logger.info(
                f"RLS θ-norm: {update_info.get('rls_theta_norm', 0):.4f} | "
                f"Integral: {update_info.get('integral_term', 0):.2f} | LastError: {update_info.get('last_error', 0):.2f}"
            )

        # Periodically save model checkpoints
        if ep % 20 == 0 or ep == episodes - 1:
            ckpt_path = os.path.join(model_dir, f"{log_prefix}_ep{ep:04d}.npz")
            agent.save(ckpt_path)
            logger.info(f"[Ep {ep:04d}] Model saved at: {ckpt_path}")

    # Save final model
    agent.save(final_model_path)
    logger.info(f"✅ Final model saved at {final_model_path}")
    writer.close()

    return rewards
