[English](README.md) | [中文](README_zh.md)

# ERL-Fill（情感驱动强化学习高精度气固流充填控制框架）

**ERL-Fill** 是一个面向高精度气固流充填控制的情感驱动强化学习框架。它融合了多模态感知、基于 Transformer 的情感建模、分阶段预训练和奖励塑造，实现了在动态工业环境下的自适应高精度控制。

## 🔍 主要亮点

- 🎯 **情感感知 RL**：基于 Transformer 的情感适配器调节探索与利用平衡。
- 📦 **多模态感知**：融合时序信号（如压力、振动）与视觉输入。
- 🧠 **分阶段预训练**：仿真→上板→真实转移，实现高效领域适应。
- 🧮 **奖励塑造**：自定义梯度奖励，优化精度与时序。
- 🔧 **模块化框架**：支持 ER-DDPG、Vanilla-DDPG、TD3、SAC、PPO、CQL、PID、TD3+BC、PPO-Lagrangian、RLS+PID。

## 📁 项目结构

```bash
.
├── baseline_experiments.py         # 基线实验脚本
├── EmotionModule.py                # 情感模块
├── experiment_runner.py            # 主实验入口
├── MutiConditionEnv.py             # 多工况环境
├── VirtualWeightController.py      # 虚拟权重控制器
├── WeightEnv.py                    # 权重环境
├── requirements.txt                # 依赖列表
├── README.md                       # 英文文档
├── README_zh.md                    # 中文文档
├── DifferentModules/               # 情感适配器、缓冲区、日志等模块
├── CommonInterface/                # 通用接口
├── analysis_outputs/               # 分析输出
├── logs/                           # 训练日志
├── runs/                           # TensorBoard 日志
├── saved_models/                   # 模型检查点
```

## ⚙️ 依赖

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

安装方式：

```bash
pip install -r requirements.txt
```

## 🚀 快速开始
ER-DDPG 基线训练示例：

```bash
python experiment_runner.py --algo er_ddpg --experiment_id 1
```
其他可选算法：ddpg, sac, ppo, cql, pid, ppo_lagrangian, td3_bc, rls_pid。

## 📖 参考文献
如使用 ERL-Fill，请引用：

```bibtex
@article{ma2025erlfill,
  title={ERL-Fill: An Emotion-Aware Reinforcement Learning Framework with Staged Pretraining for Gas--Solid Flow Filling Control},
  author={Ma, Qihang and Peng, Gaoliang and Zhang, Wei and Zhao, Benqi and Chen, Zhao and Jin, Kang},
  journal={IEEE Transactions on Industrial Electronics},
  year={2025}
}
```
## 📄 许可证
MIT License. 详见 LICENSE 文件。
