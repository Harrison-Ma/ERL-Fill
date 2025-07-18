[English](README.md) | [中文](README_zh.md)

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
├── baseline_experiments.py         # Baseline experiment script
├── ddpg_emotion_agent.pth          # Pretrained model weights
├── EmotionModule.py                # Emotion module
├── experiment_runner.py            # Main experiment entry
├── MutiConditionEnv.py             # Multi-condition environment
├── VirtualWeightController.py      # Virtual weight controller
├── WeightEnv.py                    # Weight environment
├── requirements.txt                # Dependency list
├── README.md
├── README_zh.md                    # Chinese documentation
├── DifferentModules/               # Emotion adapters, buffers, loggers, etc.
├── components/                     # Component modules
├── configs/                        # Configuration files
├── CommonInterface/                # Common interfaces
├── analysis_outputs/               # Analysis outputs
├── logs/                           # Training logs
├── runs/                           # TensorBoard logs
├── saved_models/                   # Model checkpoints
├── __pycache__/                    # Python cache files
```

## ⚙️ Dependencies

```bash
Python >= 3.8
PyTorch >= 1.12
gym
tqdm
numpy
matplotlib
scikit-learn
tensorboard
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