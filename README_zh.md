## 🌐 语言切换 / Language

- [中文](README_zh.md)
- [English](README.md)

---

# ERL-Fill

**ERL-Fill** 是一个面向高精度气固流体灌装控制的情感驱动强化学习框架。它集成了多模态感知、基于 Transformer 的情感建模、分阶段预训练和奖励塑造，能够在动态工业环境下实现自适应高精度控制。

## 🔍 亮点

- 🎯 **情感感知 RL**：基于 Transformer 的情感适配器调节探索与利用平衡。
- 📦 **多模态感知**：融合时序信号（如压力、振动）与视觉输入。
- 🧠 **分阶段预训练**：仿真→板载→真实迁移，实现高效领域适应。
- 🧮 **奖励塑造**：自定义梯度奖励，优化精度与时效。
- 🔧 **模块化框架**：支持 ER-DDPG、Vanilla-DDPG、TD3、SAC、PPO、CQL、PID、TD3+BC、PPO-Lagrangian、RLS+PID。

## 📁 项目结构

```bash
.
├── baseline_experiments.py      # 基线实验脚本
├── ddpg_emotion_agent.pth      # 预训练智能体权重
├── EmotionModule.py            # 情感建模模块
├── experiment_runner.py        # 主实验入口
├── MutiConditionEnv.py         # 多工况环境
├── README.md                   # 项目文档（英文）
├── README_zh.md                # 项目文档（中文）
├── requirements.txt            # Python 依赖
├── VirtualWeightController.py  # 虚拟称重控制器
├── WeightEnv.py                # 称重环境
├── logs/                       # 训练日志
├── runs/                       # TensorBoard 日志
├── saved_models/               # 模型检查点
├── CommonInterface/            # Modbus 与硬件接口
├── DifferentModules/           # 情感适配器、缓冲区、日志等
├── components/                 # 其他组件
├── configs/                    # 配置文件
```

## ⚙️ 依赖

```bash
Python >= 3.8
PyTorch >= 1.12
TensorBoard
gym
tqdm
numpy
pymodbus
pyserial
matplotlib
scipy
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

## 🏋️ WeightEnv 环境说明

`WeightEnv.py` 提供高精度称重控制的仿真与真实环境，支持 RL 智能体训练与评估。主要特性如下：

- **参数范围**：支持快/中/慢加速率、开度、延迟等多维参数自定义。
- **仿真与真实切换**：`use_offline_sim` 参数可选 1（仿真）、0（真实）、2（Modbus 真实）。
- **归一化与裁剪**：自动将动作归一化到实际物理范围，保证安全性。
- **奖励函数**：综合误差、时间、边界等多维奖励，支持梯度奖励与惩罚。
- **多模式支持**：可接入虚拟控制器或真实 Modbus 设备。

### 主要参数

- `target_weight`：目标称重（默认 25000g）
- `max_steps`：每回合最大步数（默认 100）
- `bounds`：各参数上下限（如加速率、开度、延迟等）

### 使用示例

```python
from WeightEnv import WeightEnv

env = WeightEnv()
state = env.reset()
action = env.normalize_action(np.random.uniform(-1, 1, env.action_dim))
next_state, reward, done, info = env.step(action)
```

更多细节请参考 `WeightEnv.py` 注释与代码实现。

## 📖 引用
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
