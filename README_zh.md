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
├── ddpg_emotion_agent.pth          # 预训练模型权重
├── EmotionModule.py                # 情感模块
├── experiment_runner.py            # 主实验入口
├── MutiConditionEnv.py             # 多工况环境
├── VirtualWeightController.py      # 虚拟权重控制器
├── WeightEnv.py                    # 权重环境
├── requirements.txt                # 依赖列表
├── README.md                       # 英文文档
├── README_zh.md                    # 中文文档
├── DifferentModules/               # 情感适配器、缓冲区、日志等模块
├── components/                     # 组件模块
├── configs/                        # 配置文件
├── CommonInterface/                # 通用接口
├── analysis_outputs/               # 分析输出
├── logs/                           # 训练日志
├── runs/                           # TensorBoard 日志
├── saved_models/                   # 模型检查点
├── __pycache__/                    # Python 缓存文件夹
```

## ⚙️ 依赖

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
