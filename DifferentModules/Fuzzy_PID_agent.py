from collections import deque
import numpy as np


class DummyMemory:
    def __init__(self):
        self.buffer = deque(maxlen=1)

    def __len__(self):
        return len(self.buffer)


class FuzzyControllerAgent:
    def __init__(self, env, device='cpu'):
        self.env = env
        self.device = device
        self.bounds = env.bounds  # dict: action_name -> (min, max)
        self.action_names = list(env.bounds.keys())

        # ✅ 添加 dummy memory 兼容 reset 函数
        self.memory = DummyMemory()
        dummy_state = [0.5, 0.5]  # 初始状态（归一化）
        dummy_action = {k: (lo + hi) / 2 for k, (lo, hi) in self.bounds.items()}
        dummy_reward = 0.0
        dummy_next_state = [0.5, 0.5]
        dummy_done = False
        self.memory.buffer.append((dummy_state, dummy_action, dummy_reward, dummy_next_state, dummy_done))

    def act(self, state):
        weight_error = state[0] if isinstance(state, (list, np.ndarray)) else state.get('weight_error', 0.0)
        time_elapsed = state[1] if isinstance(state, (list, np.ndarray)) else state.get('time', 0.0)

        base_action = 0.5 + 0.5 * np.tanh(weight_error / 10)
        time_factor = np.exp(-time_elapsed / 2.5)

        action = {}
        for name in self.action_names:
            lo, hi = self.bounds[name]
            scaled = base_action * time_factor
            act_val = lo + scaled * (hi - lo)
            action[name] = np.clip(act_val, lo, hi)

        return action

    def load(self, path):
        pass

    def save(self, path):
        pass
