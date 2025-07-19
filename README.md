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
├── EmotionModule.py                # Emotion module
├── experiment_runner.py            # Main experiment entry
├── MutiConditionEnv.py             # Multi-condition environment
├── VirtualWeightController.py      # Virtual weight controller
├── WeightEnv.py                    # Weight environment
├── requirements.txt                # Dependency list
├── README.md
├── README_zh.md                    # Chinese documentation
├── DifferentModules/               # Emotion adapters, buffers, loggers, etc.
├── CommonInterface/                # Common interfaces
├── analysis_outputs/               # Analysis outputs
├── logs/                           # Training logs
├── runs/                           # TensorBoard logs
├── saved_models/                   # Model checkpoints
```

## ⚙️ Dependencies

```bash
absl-py==2.3.1
cachetools==5.5.2
certifi==2025.7.14
charset-normalizer==3.4.2
filelock==3.16.1
fsspec==2025.3.0
google-auth==2.40.3
google-auth-oauthlib==1.0.0
grpcio==1.70.0
idna==3.10
importlib_metadata==8.5.0
Jinja2==3.1.6
Markdown==3.7
MarkupSafe==2.1.5
mpmath==1.3.0
networkx==3.1
numpy==1.24.4
nvidia-cublas-cu12==12.1.3.1
nvidia-cuda-cupti-cu12==12.1.105
nvidia-cuda-nvrtc-cu12==12.1.105
nvidia-cuda-runtime-cu12==12.1.105
nvidia-cudnn-cu12==9.1.0.70
nvidia-cufft-cu12==11.0.2.54
nvidia-curand-cu12==10.3.2.106
nvidia-cusolver-cu12==11.4.5.107
nvidia-cusparse-cu12==12.1.0.106
nvidia-nccl-cu12==2.20.5
nvidia-nvjitlink-cu12==12.9.86
nvidia-nvtx-cu12==12.1.105
oauthlib==3.3.1
pandas==2.0.3
pillow==10.4.0
protobuf==5.29.5
pyasn1==0.6.1
pyasn1_modules==0.4.2
pymodbus==3.6.9
python-dateutil==2.9.0.post0
pytz==2025.2
requests==2.32.4
requests-oauthlib==2.0.0
rsa==4.9.1
six==1.17.0
sympy==1.13.3
tensorboard==2.14.0
tensorboard-data-server==0.7.2
torch==2.4.1
torchaudio==2.4.1
torchvision==0.19.1
tqdm==4.67.1
triton==3.0.0
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