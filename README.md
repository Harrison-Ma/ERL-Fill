# About ERL-Fill

**ERL-Fill** (Emotion-aware Reinforcement Learning for Filling) is an advanced reinforcement learning framework tailored for high-precision, high-efficiency control of gas–solid two-phase flow filling systems under complex and variable working conditions.

## 🔍 Highlights

- 🎯 **Transformer-Based Emotion Adapter**: Captures temporal emotional dynamics—such as anxiety, conservativeness, and exploration—to modulate policy learning and enhance adaptability under non-stationary environments.
- 📦 **Gradient-Guided Reward Mechanism**: A multi-dimensional reward design that balances filling accuracy, timing efficiency, and operational safety, accelerating convergence while improving policy robustness.
- 🧠 **Staged Pretraining Strategy**: A three-phase training curriculum involving virtual simulation, on-board embedded control, and real-system fine-tuning (5,000 → 1,000 → 200 episodes), enabling safe and efficient policy transfer to real-world systems.
- 🧮 **Adaptive Deployment Across Diverse Conditions**: Enables robust control under varying weights, time constraints, and operational disturbances.
- 🔧 **Multi-Algorithm Benchmarking**: Supports comparative evaluation across DDPG, TD3, SAC, PPO, CQL, PID, TD3+BC, and RLS+PID.

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
Example training:

```bash
python experiment_runner.py
```
Other available algorithms: ddpg, sac, ppo, cql, pid, td3_bc, rls_pid.

| Experiment ID | Description                      |
|---------------|----------------------------------|
| 1             | Emotion mechanism ablation       |
| 2             | Baseline algorithm comparison    |
| 3             | Multi-condition testing          |
| 4             | Multi-stage pretraining          |

## 📖 Citation
If you use ERL-Fill, please cite:

```bibtex
@article{ma2025erlfill,
  author    = {Qihang Ma and Gaoliang Peng and Wei Zhang and Jinghan Wang},
  title     = {{ERL-Fill}: An Emotion-Aware Reinforcement Learning Framework with Staged Pretraining for Gas--Solid Flow Filling Control},
  journal   = {IEEE Transactions on Industrial Electronics},
  note      = {under major revision},
  year      = {2025}
}
```
📄 Paper under major revision. [Link will be provided upon publication.]

## 📄 License
MIT License. See LICENSE for details.