## 🌐 Language / 语言切换

- [English](README.md)
- [中文](README_zh.md)

---

# ERL-Fill

**ERL-Fill** is an Emotion-Driven Reinforcement Learning framework for high-precision gas–solid flow filling control. It integrates multimodal perception, transformer-based emotion modeling, staged pretraining, and reward shaping to achieve adaptive, high-accuracy performance across dynamic industrial conditions.

## 🔍 Highlights

- 🎯 **Emotion-Aware RL**: Transformer-based emotion adapter regulates exploration/exploitation balance.
- 📦 **Multimodal Sensing**: Integrates time-series signals (e.g., pressure, vibration) with visual inputs.
- 🧠 **Staged Pretraining**: Sim → Onboard → Real transfer for efficient domain adaptation.
- 🧮 **Reward Shaping**: Custom gradient-based rewards for precision and timing optimization.
- 🔧 **Modular Framework**: Supports ER-DDPG, Vanilla-DDPG, TD3, SAC, PPO, CQL, PID, TD3+BC, PPO-Lagrangian, and RLS+PID.

## 📁 Project Structure

```bash
.
├── baseline_experiments.py      # Baseline experiment scripts
├── ddpg_emotion_agent.pth      # Pretrained agent weights
├── EmotionModule.py            # Emotion modeling module
├── experiment_runner.py        # Main experiment launcher
├── MutiConditionEnv.py         # Multi-condition environment
├── README.md                   # Project documentation (English)
├── README_zh.md                # Project documentation (Chinese)
├── requirements.txt            # Python dependencies
├── VirtualWeightController.py  # Virtual weight controller
├── WeightEnv.py                # Weight control environment
├── logs/                       # Training logs
├── runs/                       # TensorBoard logs
├── saved_models/               # Model checkpoints
├── CommonInterface/            # Modbus and hardware interfaces
├── DifferentModules/           # Emotion adapters, buffers, loggers, etc.
├── components/                 # Additional components
├── configs/                    # Configuration files
```

## ⚙️ Dependencies

```bash
Python >= 3.8

PyTorch >= 1.12

TensorBoard

Gym

Tqdm

Numpy

pymodbus


# For hardware/Modbus support:
pyserial


# For plotting/analysis (optional):
matplotlib

scipy
```

Install via:

```bash
pip install -r requirements.txt
```
## 🚀 Getting Started
Example training (ER-DDPG baseline):

```bash
python experiment_runner.py --algo er_ddpg --experiment_id 1
```
Other available algorithms: ddpg, sac, ppo, cql, pid, ppo_lagrangian, td3_bc, rls_pid.

## 🏋️ WeightEnv Environment Overview

`WeightEnv.py` provides a high-precision weight control simulation and real environment for RL agent training and evaluation. Key features include:

- **Parameter Ranges**: Supports multi-dimensional parameters such as fast/medium/slow feed rates, valve openings, and delays.
- **Simulation & Real Modes**: Switch between simulation (`use_offline_sim=1`), real device (`use_offline_sim=0`), and Modbus real (`use_offline_sim=2`).
- **Normalization & Clipping**: Actions are automatically normalized and clipped to safe physical ranges.
- **Reward Function**: Multi-dimensional reward considering error, time, and boundary penalties, with gradient-based shaping.
- **Multi-mode Support**: Can connect to a virtual controller or real Modbus device.

### Main Parameters

- `target_weight`: Target weight (default 25000g)
- `max_steps`: Max steps per episode (default 100)
- `bounds`: Parameter ranges (feed rates, openings, delays, etc.)

### Usage Example

```python
from WeightEnv import WeightEnv

env = WeightEnv()
state = env.reset()
action = env.normalize_action(np.random.uniform(-1, 1, env.action_dim))
next_state, reward, done, info = env.step(action)
```

See `WeightEnv.py` for more details and code comments.

## 📖 Citation
If you use ERL-Fill, please cite:

```bibtex
@article{ma2025erlfill,
  title={ERL-Fill: An Emotion-Aware Reinforcement Learning Framework with Staged Pretraining for Gas--Solid Flow Filling Control},
  author={Ma, Qihang and Peng, Gaoliang and Zhang, Wei and Zhao, Benqi and Chen, Zhao and Jin, Kang},
  journal={IEEE Transactions on Industrial Electronics},
  year={2025}
}
```
## 📄 License
MIT License. See LICENSE for details.