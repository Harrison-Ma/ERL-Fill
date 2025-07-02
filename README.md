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
├── rl-psa_v1.0/                # Main training framework
│   ├── agents/                 # Reinforcement learning agents
│   ├── envs/                   # Gas–solid filling environments
│   ├── DifferentModule/        # Emotion adapters, buffers, loggers, etc.
│   ├── experiment_runner.py    # Main experiment launcher
│   └── config.py               # Global settings
├── saved_models/              # Model checkpoints
├── runs/                      # TensorBoard logs
├── logs/                      # Training logs
└── README.md
```

## ⚙️ Dependencies

```bash
Python ≥ 3.8

PyTorch ≥ 1.12

TensorBoard

gym

tqdm

numpy



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