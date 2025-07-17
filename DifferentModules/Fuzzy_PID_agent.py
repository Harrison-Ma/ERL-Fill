from collections import deque
import numpy as np


class DummyMemory:
    """
    A minimal dummy memory buffer to maintain compatibility with interfaces
    expecting a memory object with a buffer attribute and length.
    It stores at most one experience tuple.
    """
    def __init__(self):
        """Initialize the dummy buffer with a maximum size of 1."""
        self.buffer = deque(maxlen=1)

    def __len__(self):
        """Return the current number of stored experience tuples."""
        return len(self.buffer)


class FuzzyControllerAgent:
    """
    A simple fuzzy controller agent for the environment that generates
    actions based on a heuristic combining weight error and elapsed time.

    Attributes:
        env: The environment instance, expected to have action bounds.
        device: Computation device ('cpu' or 'cuda').
        bounds: Dictionary mapping action names to (min, max) ranges.
        action_names: List of action names.
        memory: DummyMemory instance for compatibility with training loops.
    """

    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.bounds = env.bounds  # dict: action_name -> (min, max)
        self.action_names = list(env.bounds.keys())

        # Initialize dummy memory with one default experience tuple for compatibility
        self.memory = DummyMemory()
        dummy_state = [0.5, 0.5]  # Initial normalized state
        dummy_action = {k: (lo + hi) / 2 for k, (lo, hi) in self.bounds.items()}  # Midpoint actions
        dummy_reward = 0.0
        dummy_next_state = [0.5, 0.5]
        dummy_done = False
        self.memory.buffer.append((dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done))

    def act(self, state):
        """
        Generate an action based on the current state using a fuzzy heuristic.

        Args:
            state (list, np.ndarray, or dict): Current environment state. Expected to contain
                'weight_error' and 'time' values if dict, or as elements if list/array.

        Returns:
            dict: Dictionary mapping action names to their computed values,
                  clipped within their bounds.
        """
        # Extract relevant features from state
        weight_error = state[0] if isinstance(state, (list, np.ndarray)) else state.get('weight_error', 0.0)
        time_elapsed = state[1] if isinstance(state, (list, np.ndarray)) else state.get('time', 0.0)

        # Compute base action component (sigmoid-like scaling of weight error)
        base_action = 0.5 + 0.5 * np.tanh(weight_error / 10)
        # Decay factor based on elapsed time to reduce action magnitude over time
        time_factor = np.exp(-time_elapsed / 2.5)

        action = {}
        for name in self.action_names:
            lo, hi = self.bounds[name]
            scaled = base_action * time_factor
            # Scale and clip action within bounds
            act_val = lo + scaled * (hi - lo)
            action[name] = np.clip(act_val, lo, hi)

        return action

    def load(self, path):
        """
        Placeholder for loading agent parameters from a file.

        Args:
            path (str): File path to load from.
        """
        pass

    def save(self, path):
        """
        Placeholder for saving agent parameters to a file.

        Args:
            path (str): File path to save to.
        """
        pass

