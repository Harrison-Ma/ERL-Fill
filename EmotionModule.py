# import SysOut

from torch.nn import TransformerEncoder, TransformerEncoderLayer

import numpy as np

import torch
import torch.nn as nn

from collections import deque


class EmotionTransformer(nn.Module):
    def __init__(self, input_dim=3, d_model=32, nhead=4, num_layers=4, dropout=0.3, device='cpu', max_len=20):
        super(EmotionTransformer, self).__init__()

        self.device = device
        self.max_len = max_len
        self.embedding = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.randn(max_len, d_model))  # learnable
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout, batch_first=False)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, input_dim)
        self.history = deque(maxlen=max_len)

        self.to(self.device)

    def forward(self, emotion_sequence):
        x = emotion_sequence.permute(1, 0, 2)  # [seq_len, 1, 3]
        x = self.embedding(x)
        x = x + self.positional_encoding[:x.shape[0]].unsqueeze(1).to(self.device)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[-1])
        return torch.tanh(x) * 0.5 + 0.5
        # return torch.sigmoid(x)                       # 限制在 [0,1]

    def update(self, new_emotion):
        tensor = torch.tensor(new_emotion, dtype=torch.float32)
        self.history.append(tensor)

    def get_state(self):
        if len(self.history) == 0:
            return torch.FloatTensor([[0.5, 0.5, 0.0]]).unsqueeze(0).to(self.device)
        padded = [self.history[0].cpu().numpy()] * (self.history.maxlen - len(self.history)) + \
                 [h.cpu().numpy() for h in self.history]
        padded_array = np.array(padded, dtype=np.float32)
        tensor_seq = torch.from_numpy(padded_array).unsqueeze(0).to(self.device)  # [1, seq_len, 3]
        return tensor_seq


class EmotionModule:
    def __init__(self, device='cpu'):
        self.transformer = EmotionTransformer(device=device)
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.device = device

    def update(self, reward, movement):
        reward_term = 1 - np.tanh(reward)
        move_magnitude = np.linalg.norm(movement)
        move_factor = np.clip(move_magnitude, 0.0, 1.0)

        # exploration 更依赖惩罚项，突出 early stage 的变动
        exploration = 0.8 * reward_term + 0.2 * move_factor

        # conservativeness 以奖励为主，但加入轻微扰动
        conservativeness = np.clip(np.tanh(reward) - 0.1 * move_factor, 0.0, 1.0)

        # anxiety 下降更快，并引入 clip 控制
        scaled_reward = np.tanh(reward * 0.001)
        anxiety_delta = -0.2 * scaled_reward + 0.05 * move_factor
        anxiety = self.current_emotion[2] + anxiety_delta
        anxiety = np.clip(anxiety, 0.0, 1.0)

        # 更新情绪，增强响应速率（0.8）
        new_emotion = np.clip([
            0.8 * self.current_emotion[0] + 0.2 * exploration,
            0.8 * self.current_emotion[1] + 0.2 * conservativeness,
            anxiety
        ], 0.0, 1.0)

        self.current_emotion = new_emotion
        self.transformer.update(new_emotion)
        return self.current_emotion.copy()

    def get_emotion(self):
        with torch.no_grad():
            sequence = self.transformer.get_state()
            predicted = self.transformer(sequence)
            return predicted.squeeze(0).cpu().numpy()

    def save(self, path):
        torch.save(self.transformer.state_dict(), path)

    def load(self, path):
        state_dict = torch.load(path, map_location=self.device)
        self.transformer.load_state_dict(state_dict)

def save_emotion_module(emotion_module, path):
    """保存 EmotionModule 的 transformer 权重"""
    torch.save(emotion_module.transformer.state_dict(), path)

def load_emotion_module(emotion_module, path, device='cpu'):
    """加载 EmotionModule 的 transformer 权重"""
    state_dict = torch.load(path, map_location=device)
    emotion_module.transformer.load_state_dict(state_dict)

class EmotionTransformerV2(nn.Module):
    def __init__(self, input_dim=3, d_model=16, nhead=2, num_layers=2, dropout=0.2, device='cpu'):
        super(EmotionTransformerV2, self).__init__()
        self.device = device
        self.embedding = nn.Linear(input_dim, d_model)
        encoder_layer = TransformerEncoderLayer(d_model=d_model, nhead=nhead, dropout=dropout)
        self.transformer_encoder = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.fc_out = nn.Linear(d_model, input_dim)
        self.history = deque(maxlen=20)

        self.to(self.device)

    def forward(self, emotion_sequence):
        """
        emotion_sequence: shape [1, seq_len, 3]
        """
        x = emotion_sequence.permute(1, 0, 2)  # [1, seq_len, 3] → [seq_len, 1, 3]
        x = self.embedding(x)
        x = self.transformer_encoder(x)
        x = self.fc_out(x[-1])         # [1, 3]
        x = torch.tanh(x) * 0.5 + 0.5  # 限制输出在 [0, 1]，保留激活活跃性
        return x

    def update(self, new_emotion):
        tensor = torch.tensor(new_emotion, dtype=torch.float32)
        self.history.append(tensor)

    def get_state(self):
        if len(self.history) == 0:
            return torch.FloatTensor([[0.5, 0.5, 0.0]]).unsqueeze(0).to(self.device)

        padded = [self.history[0].cpu().numpy()] * (self.history.maxlen - len(self.history)) + \
                 [h.cpu().numpy() for h in self.history]
        padded_array = np.array(padded, dtype=np.float32)
        tensor_seq = torch.from_numpy(padded_array).unsqueeze(0).to(self.device)
        return tensor_seq

class EmotionModuleSimple:
    """
    简化情绪模块：用于对照实验，模拟低效模糊感知情绪。
    特征：
    - 更新反应慢；
    - 加入噪声扰动；
    - 抑制 reward 驱动；
    """

    def __init__(self):
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)
        self.alpha = 0.05  # EMA 权重（更慢）
        self.noise_scale = 0.02  # 噪声扰动

    def update(self, reward, movement):
        """
        简化更新机制 + 噪声扰动，抑制 reward 主导行为。
        """
        reward_term = 1.0 - np.tanh(reward)  # exploration 倾向（抑制高 reward）
        move_magnitude = np.clip(np.linalg.norm(movement), 0.0, 1.0)

        # === 构造情绪输入：用较小系数淡化变化 ===
        exploration = 0.5 * reward_term + 0.0 * move_magnitude
        conservativeness = np.clip(0.3 * np.tanh(reward) - 0.0 * move_magnitude, 0.0, 1.0)
        anxiety_delta = -0.05 * np.tanh(reward * 0.001) + 0.00 * move_magnitude
        anxiety = np.clip(self.current_emotion[2] + anxiety_delta, 0.0, 1.0)

        target_emotion = np.array([exploration, conservativeness, anxiety], dtype=np.float32)

        # === EMA 叠加微小高斯扰动，使行为更模糊 ===
        noise = np.random.normal(0, self.noise_scale, size=3).astype(np.float32)
        target_emotion = np.clip(target_emotion + noise, 0.0, 1.0)

        new_emotion = (1 - self.alpha) * self.current_emotion + self.alpha * target_emotion
        self.current_emotion = np.clip(new_emotion, 0.0, 1.0)

        return self.current_emotion.copy()

    def get_emotion(self):
        return self.current_emotion.copy()

class EmotionModuleNone:
    """
    无情感调节模块（对照组用）：
    始终返回固定的中性情绪状态 [0.5, 0.5, 0.0]。
    不随 reward 或 state 变化更新。
    """
    def __init__(self, device='cpu'):
        self.device = device
        self.current_emotion = np.array([0.5, 0.5, 0.0], dtype=np.float32)

    def update(self, reward, movement):
        # 不进行任何更新
        return self.current_emotion.copy()

    def get_emotion(self):
        return self.current_emotion.copy()

