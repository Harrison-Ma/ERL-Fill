import numpy as np
import torch
import torch.nn as nn
from collections import deque
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=32, nhead=4, num_layers=4, dropout=0.3, device='cpu', max_len=20):
        """
        Transformer-based module for processing emotional state sequences.

        Args:
            input_dim (int): Dimension of each emotion vector (default: 3).
            d_model (int): Dimension of the Transformer model (embedding size).
            nhead (int): Number of attention heads in the Transformer.
            num_layers (int): Number of TransformerEncoder layers.
            dropout (float): Dropout rate.
            device (str): Computation device ('cpu' or 'cuda').
            max_len (int): Maximum length of the emotion history sequence.
        """
        super(EmotionTransformer, self).__init__()

        self.device = device
        self.max_len = max_len

        # Linear layer to embed input emotion vectors into model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Learnable positional encoding
        self.positional_encoding = nn.Parameter(torch.randn(max_len, d_model))

        # Transformer encoder configuration
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False
        )
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Final linear projection to return emotion-like output (dim = input_dim)
        self.fc_out = nn.Linear(d_model, input_dim)

        # Emotion history buffer (fixed-length deque)
        self.history = deque(maxlen=max_len)

        # Move model to the specified device
        self.to(self.device)

    def forward(self, emotion_sequence):
        """
        Forward pass of the EmotionTransformer.

        Args:
            emotion_sequence (Tensor): Input tensor of shape [batch, seq_len, input_dim].

        Returns:
            Tensor: Output emotion prediction of shape [input_dim],
                    normalized to [0, 1] using a scaled Tanh.
        """
        # Rearrange to [seq_len, batch, input_dim] as required by Transformer
        x = emotion_sequence.permute(1, 0, 2)  # [seq_len, 1, 3]

        # Embed input and add positional encoding
        x = self.embedding(x)
        x = x + self.positional_encoding[:x.shape[0]].unsqueeze(1).to(self.device)

        # Transformer encoding
        x = self.transformer_encoder(x)

        # Final output projection + scaled Tanh normalization
        x = self.fc_out(x[-1])  # Use last token representation
        return torch.tanh(x) * 0.5 + 0.5  # Output in range [0, 1]

    def update(self, new_emotion):
        """
        Add a new emotion vector to the history buffer.

        Args:
            new_emotion (array-like): New emotion vector (length = input_dim).
        """
        tensor = torch.tensor(new_emotion, dtype=torch.float32)
        self.history.append(tensor)

    def get_state(self):
        """
        Get the current emotion history as a padded tensor.

        Returns:
            Tensor: Emotion sequence tensor of shape [1, seq_len, input_dim],
                    ready to be passed to the model.
        """
        if len(self.history) == 0:
            # Default neutral state if no history is available
            return torch.FloatTensor([[0.5, 0.5, 0.0]]).unsqueeze(0).to(self.device)

        # Pad with the first entry if not full
        padded = [self.history[0].cpu().numpy()] * (self.history.maxlen - len(self.history)) + \
                 [h.cpu().numpy() for h in self.history]
        padded_array = np.array(padded, dtype=np.float32)

        # Return tensor in shape [1, seq_len, input_dim]
        tensor_seq = torch.from_numpy(padded_array).unsqueeze(0).to(self.device)
        return tensor_seq


class EmotionModule:
    def __init__(self, device='cpu'):
        """
        EmotionModule maintains and updates a 3-dimensional emotion vector
        based on agent-environment interaction signals (reward and movement).

        It integrates a Transformer-based temporal emotion encoder for
        emotion sequence modeling.

        Args:
            device (str): Computation device, e.g., 'cpu' or 'cuda'.
        """
        self.transformer = EmotionTransformer(device=device)
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)  # [exploration, conservativeness, anxiety]
        self.device = device

    def update(self, reward, movement):
        """
        Update the internal emotional state based on recent reward and movement.

        Emotion dimensions:
            - Exploration: encourages trying new actions (driven by low rewards and movement).
            - Conservativeness: prefers stability (driven by good rewards, penalized by rapid movement).
            - Anxiety: accumulates stress, rises under uncertainty.

        Args:
            reward (float): Immediate reward from the environment.
            movement (array-like): Recent movement vector (e.g., ∆state or action norm).

        Returns:
            np.ndarray: Updated 3-dimensional emotion vector.
        """
        # Compute exploration as a weighted combination of low reward and movement intensity
        reward_term = 1 - np.tanh(reward)
        move_magnitude = np.linalg.norm(movement)
        move_factor = np.clip(move_magnitude, 0.0, 1.0)

        exploration = 0.8 * reward_term + 0.2 * move_factor

        # Compute conservativeness primarily from reward, slightly reduced by movement
        conservativeness = np.clip(np.tanh(reward) - 0.1 * move_factor, 0.0, 1.0)

        # Compute anxiety as an accumulated signal of stress
        scaled_reward = np.tanh(reward * 0.001)
        anxiety_delta = -0.2 * scaled_reward + 0.05 * move_factor
        anxiety = self.current_emotion[2] + anxiety_delta
        anxiety = np.clip(anxiety, 0.0, 1.0)

        # Update current emotion vector with smoothing (moving average)
        new_emotion = np.clip([
            0.8 * self.current_emotion[0] + 0.2 * exploration,
            0.8 * self.current_emotion[1] + 0.2 * conservativeness,
            anxiety
        ], 0.0, 1.0)

        self.current_emotion = new_emotion
        self.transformer.update(new_emotion)
        return self.current_emotion.copy()

    def get_emotion(self):
        """
        Predict the emotion output using the internal Transformer model
        based on historical emotion sequence.

        Returns:
            np.ndarray: Predicted emotion vector of shape (3,), range [0, 1].
        """
        with torch.no_grad():
            sequence = self.transformer.get_state()
            predicted = self.transformer(sequence)
            return predicted.squeeze(0).cpu().numpy()

    def save(self, path):
        """
        Save the Transformer model weights to disk.

        Args:
            path (str): File path to save the model.
        """
        torch.save(self.transformer.state_dict(), path)

    def load(self, path):
        """
        Load Transformer model weights from disk.

        Args:
            path (str): File path of the saved model.
        """
        state_dict = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(state_dict)


def save_emotion_module(emotion_module, path):
    """
    Save the internal Transformer weights of an EmotionModule instance.

    Args:
        emotion_module (EmotionModule): The emotion module whose transformer is to be saved.
        path (str): Path to save the model weights.
    """
    torch.save(emotion_module.transformer.state_dict(), path)


def load_emotion_module(emotion_module, path, device='cpu'):
    """
    Load the Transformer weights into an EmotionModule instance.

    Args:
        emotion_module (EmotionModule): The emotion module whose transformer is to be loaded.
        path (str): Path of the saved model weights.
        device (str): Device to map the loaded weights to.
    """
    state_dict = torch.load(path, map_location=device)
    emotion_module.transformer.load_state_dict(state_dict)


class EmotionTransformerV2(nn.Module):
    def __init__(self, input_dim=3, d_model=16, nhead=2, num_layers=2, dropout=0.2, device='cpu'):
        """
        A lightweight Transformer-based model for modeling temporal emotion sequences.

        Args:
            input_dim (int): Dimension of emotion input (default: 3).
            d_model (int): Dimensionality of Transformer embeddings.
            nhead (int): Number of attention heads.
            num_layers (int): Number of Transformer encoder layers.
            dropout (float): Dropout rate in Transformer layers.
            device (str): Device to run the model on.
        """
        super(EmotionTransformerV2, self).__init__()
        self.device = device

        # Linear embedding layer to project input to d_model dimension
        self.embedding = nn.Linear(input_dim, d_model)

        # Transformer encoder
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output projection back to emotion space
        self.fc_out = nn.Linear(d_model, input_dim)

        # Historical buffer for temporal emotion states
        self.history = deque(maxlen=20)

        # Move model to the specified device
        self.to(self.device)

    def forward(self, emotion_sequence):
        """
        Forward pass of the transformer model.

        Args:
            emotion_sequence (Tensor): Input sequence of shape [1, seq_len, 3]

        Returns:
            Tensor: Output emotion vector of shape [1, 3], values in range [0, 1].
        """
        # Permute to [seq_len, 1, 3] for transformer processing
        x = emotion_sequence.permute(1, 0, 2)

        # Project to transformer embedding space
        x = self.embedding(x)

        # Apply transformer encoder
        x = self.transformer_encoder(x)

        # Use the last time step's output and project to emotion space
        x = self.fc_out(x[-1])  # [1, 3]

        # Squash output into [0, 1] using scaled tanh activation
        x = torch.tanh(x) * 0.5 + 0.5
        return x

    def update(self, new_emotion):
        """
        Update the emotion history with a new emotion vector.

        Args:
            new_emotion (array-like): A new emotion vector to append (shape: [3]).
        """
        tensor = torch.tensor(new_emotion, dtype=torch.float32)
        self.history.append(tensor)

    def get_state(self):
        """
        Get the current emotion sequence state as a tensor suitable for transformer input.

        Returns:
            Tensor: A padded tensor of shape [1, seq_len, 3] on the correct device.
        """
        if len(self.history) == 0:
            # Return a neutral emotion state if history is empty
            return torch.FloatTensor([[0.5, 0.5, 0.0]]).unsqueeze(0).to(self.device)

        # Pad sequence with the first emotion if history is shorter than max length
        padded = [self.history[0].cpu().numpy()] * (self.history.maxlen - len(self.history)) + \
                 [h.cpu().numpy() for h in self.history]

        padded_array = np.array(padded, dtype=np.float32)
        tensor_seq = torch.from_numpy(padded_array).unsqueeze(0).to(self.device)
        return tensor_seq


class EmotionModuleSimple:
    """
    Simplified emotion module for ablation or baseline experiments.
    Characteristics:
    - Responds slowly to input changes (low EMA weight)
    - Adds Gaussian noise to simulate fuzzy affect perception
    - Suppresses strong reward-driven emotional responses
    """

    def __init__(self):
        # Initial neutral emotion state: [exploration, conservativeness, anxiety]
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)

        self.alpha = 0.05  # Exponential Moving Average (EMA) weight — slow response
        self.noise_scale = 0.02  # Noise scale for Gaussian perturbation

    def update(self, reward, movement):
        """
        Updates the internal emotion state based on reward and movement inputs,
        with added noise and weakened sensitivity to changes.

        Args:
            reward (float): The reward signal from the environment.
            movement (array-like): The action or change vector representing movement.

        Returns:
            np.ndarray: The updated emotion vector (3D).
        """
        reward_term = 1.0 - np.tanh(reward)  # Suppresses high-reward responses
        move_magnitude = np.clip(np.linalg.norm(movement), 0.0, 1.0)

        # Emotion signals — designed to change slowly and weakly
        exploration = 0.5 * reward_term
        conservativeness = np.clip(0.3 * np.tanh(reward), 0.0, 1.0)
        anxiety_delta = -0.05 * np.tanh(reward * 0.001)
        anxiety = np.clip(self.current_emotion[2] + anxiety_delta, 0.0, 1.0)

        target_emotion = np.array([exploration, conservativeness, anxiety], dtype=np.float32)

        # Add Gaussian noise for fuzziness
        noise = np.random.normal(0, self.noise_scale, size=3).astype(np.float32)
        target_emotion = np.clip(target_emotion + noise, 0.0, 1.0)

        # EMA update for slow response
        new_emotion = (1 - self.alpha) * self.current_emotion + self.alpha * target_emotion
        self.current_emotion = np.clip(new_emotion, 0.0, 1.0)

        return self.current_emotion.copy()

    def get_emotion(self):
        """
        Returns the current emotion vector.

        Returns:
            np.ndarray: Current internal emotion (shape: [3]).
        """
        return self.current_emotion.copy()


class EmotionModuleNone:
    """
    Emotion-less module for control experiments.
    Always returns a fixed neutral emotion vector.
    Ignores all updates or external feedback.

    Use case: acts as a baseline with no emotion influence.
    """

    def __init__(self, device='cpu'):
        self.device = device
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    def update(self, reward, movement):
        """
        No-op update function. Emotion is fixed and not influenced by input.

        Args:
            reward (float): Ignored.
            movement (array-like): Ignored.

        Returns:
            np.ndarray: Constant emotion vector [0.5, 0.5, 0.0].
        """
        return self.current_emotion.copy()

    def get_emotion(self):
        """
        Returns the constant neutral emotion state.

        Returns:
            np.ndarray: Fixed emotion vector [0.5, 0.5, 0.0].
        """
        return self.current_emotion.copy()
