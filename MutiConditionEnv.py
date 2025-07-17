import numpy as np
import random
import time
from collections import deque
from pymodbus.client import ModbusSerialClient

from VirtualWeightController import VirtualWeightController

import platform
if platform.system() == 'Windows':
    port_name = 'COM4'
else:
    port_name = '/dev/ttyUSB0'

# === 实验配置表 ===
experiment_configs = {
    "variant_25kg": {
        "name": "Condition Variant - 25kg±25g",
        "env_kwargs": {"target_weight": 25000, "target_weight_err": 25, "target_time": 2.5},
    },
    "variant_20kg": {
        "name": "Condition Variant - 20kg±20g",
        "env_kwargs": {"target_weight": 20000, "target_weight_err": 20, "target_time": 2},
    },
    "variant_15kg": {
        "name": "Condition Variant - 15kg±15g",
        "env_kwargs": {"target_weight": 15000, "target_weight_err": 15, "target_time": 1.5},
    }
}

# 环境定义（保持不变），默认为25kg的环境
class WeightEnv:
    def __init__(self):
        self.target_weight = 25000
        self.target_weight_err = 25
        self.target_time = 2.5

        self.controller = VirtualWeightController()

        self.episode_counter = 0  # 当前批次编号
        self.step_counter = 0  # 当前批次的步数

        self.max_steps = 100  # 默认最大步数

        self.recent_errors = deque(maxlen=int(self.max_steps/5))
        # 参数取值范围（上下限）
        self.bounds = {
            "fast_weight": (7000, 18000),# 快加速率范围 (g/s)
            "medium_weight": (0, 1),    # 中加速率范围 (g/s)
            "slow_weight": (24900, 25000),    # 慢加速率范围 (g/s)

            "fast_opening": (35, 75),        # 快加开度范围 (%)
            "medium_opening": (3, 5),      # 中加开度范围 (%)
            "slow_opening": (5, 20),          # 慢加开度范围 (%)

            "fast_delay": (100, 300),         # 快加延迟范围 (ms)
            "medium_delay": (100, 200),       # 中加延迟范围 (ms)
            "slow_delay": (100, 200),         # 慢加延迟范围 (ms)

            "unload_delay": (300, 500)         # 卸料时间范围 (ms)
        }
        self._update_scaling_params()

        self.scale_centers = {}
        self.scale_ranges = {}
        for key, (low, high) in self.bounds.items():
            self.scale_centers[key] = (high + low) / 2
            self.scale_ranges[key] = (high - low) / 2

        self.state_dim = 2 #每次输出的称重过程结果，例如误差、时间、速率等
        self.action_dim = len(self.bounds) #就是每次称重的输入控制变量，例如快加目标值、中加目标值、慢加目标值以及开度、延时等。

        # 初始状态
        self.state = None

        self.agent = None  # ✅ 预留 agent 接口

        self.use_offline_sim = 1

        self.target_weight = 25000  # 目标重量100kg
        self.target_weight_err = 25  # 目标重量100kg
        self.target_time = 2
        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.material_coefficient = 1.0  # 材料系数，假设为1.0
        self.registers = [0] * 300  # 模拟Modbus寄存器

        # 单位为开度/秒
        # 当前速度
        self.current_speed = 0
        # 电机速度
        self.motor_speed = 10000
        # 电机加速度
        self.motor_acceleration = 5000000
        # 电机减速度
        self.motor_deceleration = 100000
        # 电机方向
        self.motor_direction = 0
        self.speed = 0

        self.client = ModbusSerialClient(
            port=port_name,
            baudrate=460800,
            parity='E',
            timeout=3
        )

        if self.use_offline_sim == 0:
            if self.client.connect():
                print("Successfully connected to Modbus device")
            else:
                print("Failed to connect")

    def _update_scaling_params(self):
        self.scale_centers = {}
        self.scale_ranges = {}
        for key, (low, high) in self.bounds.items():
            self.scale_centers[key] = (high + low) / 2
            self.scale_ranges[key] = (high - low) / 2

    @staticmethod
    def default_config_25kg():
        return {
            "target_weight": 25000,
            "target_weight_err": 25,
            "target_time": 2.5
        }

    def normalize_action(self, action):
        """将网络输出的[-1,1]动作映射到实际参数范围"""
        scaled_action = {}
        for idx, key in enumerate(self.bounds):
            scaled_action[key] = action[idx] * self.scale_ranges[key] + self.scale_centers[key]
        return scaled_action

    def attach_agent(self, agent):
        self.agent = agent  # ✅ 接收 agent（用于访问其 memory）

    def reset(self):
        """重置环境并返回归一化后的初始状态向量"""

        # ✅ 初始化状态向量
        if self.episode_counter == 0 or self.agent is None or len(self.agent.memory) < 10:
            # 首轮或经验不足 → 随机初始化并归一化
            raw_state = np.array([
                random.uniform(0, self.target_weight_err),
                random.uniform(0, self.target_time)
            ], dtype=np.float32)
        else:
            # 否则 → 采样经验池并均值（经验中已归一化，无需再除）
            # samples = random.sample(self.agent.memory, 10)
            samples = random.sample(self.agent.memory.buffer, 10)
            # print("samples:", samples)
            raw_state = np.mean([s[0] for s in samples], axis=0).astype(np.float32)

        # ✅ 状态归一化（如果已归一化过则此操作可略）
        state_vector = np.array([
            raw_state[0] / self.target_weight_err,
            raw_state[1] / self.target_time
        ], dtype=np.float32)

        # ✅ 重置环境状态变量
        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.step_counter = 0
        self.episode_counter += 1
        self.recent_errors.clear()

        return state_vector

    def _state_to_vector(self):
        """将字典状态转换为向量"""
        return np.array([self.state[key] for key in self.param_order], dtype=np.float32)

    def write_register(self, address, value):
        self.registers[address] = value

    def write_registers(self, address, values):
        for i, val in enumerate(values):
            self.registers[address + i] = val

    def read_holding_registers(self, address, count):
        return self.registers[address:address + count]

    # def motor_simulation(self, target_pos):
    #     # 简化的电机模拟：直接设置速度为位置差的1%
    #     self.current_speed = abs(target_pos - self.registers[10]) * 0.01

    def motor_simulation(self, target_pos):
        # time.sleep(100/1000)
        # global motor_speed, speed, current_speed, motor_acceleration, current_opening, motor_deceleration, motor_direction
        # 每一毫秒更新一次

        # print("target_pos ", target_pos)
        # print("self.current_opening ", self.current_opening)

        # 计算减速需要时间
        decelerate_time = self.current_speed / self.motor_deceleration
        # print("减速需要时间",decelerate_time)
        # 计算减速需要位移
        decelerate_distance = 0.5 * self.motor_deceleration * np.power(decelerate_time, 2)
        # print("减速需要位移",decelerate_distance)
        if abs(self.current_opening - target_pos) > 1:
            # 更新速度
            if self.current_opening < target_pos:
                # 正转
                self.motor_direction = 1
                if self.current_opening < target_pos - decelerate_distance:
                    # 加速阶段
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                        # print("self.motor_acceleration", self.motor_acceleration)
                        # print("self.current_speed",self.current_speed)
                        # print("加速")
                else:
                    # 减速阶段
                    self.current_speed -= self.motor_deceleration / 1000
                    # print("减速")
            elif self.current_opening > target_pos:
                # 反转
                self.motor_direction = -1
                if self.current_opening > decelerate_distance:
                    # 加速阶段
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                        # print("加速")
                else:
                    # 减速阶段
                    self.current_speed -= self.motor_deceleration / 1000
                    # print("减速")
        else:
            self.current_speed = 0
            self.current_opening = target_pos

        self.speed = self.current_speed * self.motor_direction
        self.current_opening += self.speed / 1000

        # print("self.speed / 1000 ", self.current_speed)

        # print("self.current_opening ", self.current_opening)
        # # print(current_opening)
        # return current_opening

    def weight_simulation(self, opening, material_coeff):
        random_number = random.uniform(-1, 1)
        # 简化的重量模拟：重量增加 = 开度 * 材料系数 * 时间间隔
        # 假设时间间隔为1ms（0.001秒）
        self.current_weight = int(self.current_weight + opening * (material_coeff + random_number))

    # def weight_simulation(opening, coefficient):
    #     global current_weight
    #     random_number = random.uniform(-1, 1)
    #     # print("随机数:", random_number*current_opening*0.1)
    #     # 更新当前重量，增加随机数
    #     # 随机数根据开度变化
    #     # print("add value:",opening*coefficient+ random_number*opening)
    #     # print("add_random:",random_number*opening)
    #     current_weight = int(current_weight + opening * (coefficient + random_number))

    def _get_current_opening(self):
        # 根据阶段返回对应开度（简化逻辑）
        if self.current_weight < 50000:  # 快加阶段
            return self.state["fast_opening"]
        elif self.current_weight < 80000:  # 中加阶段
            return self.state["medium_opening"]
        else:  # 慢加阶段
            return self.state["slow_opening"]

    def _run_simulation(self, action):
        # 写入控制参数到寄存器
        Arg = [
            self.target_weight,
            int(action["fast_weight"]),
            int(action["medium_weight"]),
            int(action["slow_weight"]),
            int(action["fast_delay"]),
            int(action["medium_delay"]),
            int(action["slow_delay"]),
            int(action["fast_opening"]),
            int(action["medium_opening"]),
            int(action["slow_opening"]),
            int(action["unload_delay"])
        ]
        print("Arg:",Arg)
        self.client.write_registers(200, Arg)

        # 开启启动信号
        # 读取当前寄存器值（假设是保持寄存器或输入寄存器）
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        # 设置第0位为1
        new_value = current_value | (1 << 0)
        # 将新的值写回寄存器
        self.client.write_register(70, new_value)

        # 开启允加信号
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        # 设置第1位为1
        new_value = current_value | (1 << 1)
        # 将新的值写回寄存器
        self.client.write_register(70, new_value)

        # 统计时间
        start_time = time.time()
        self.client.write_registers(5, 0)  # 状态复位

        total_time = 0
        speeds, openings, weights = [], [], []
        PackFinish = 0
        self.previous_time = time.time()
        while not PackFinish:
            self.current_time = time.time()
            self.elapsed_time = self.current_time - self.previous_time
            self.previous_time = self.current_time
            self.elapsed_time = round(self.elapsed_time * 1000)

            # time.sleep(0.003)
            controler_modbus_reg = self.client.read_holding_registers(address=0, count=124)  # 读取控制板数据
            target_status = controler_modbus_reg.registers[6]  # 获取控制板当前运行状态
            # print("target_status",target_status)
            PackFinish = (target_status & (1 << 5))  # 定值信号
            # print("PackFinish ",PackFinish)

            # if (self.current_opening < 0):
            #     PackFinish = 1
            # print("PackFinish ", PackFinish)

            target_pos = controler_modbus_reg.registers[10]  # 获取目标位置
            # print("target_pos ",target_pos)
            # if self.current_weight>24000:
                # print("bbb")

            if self.elapsed_time < 200:
                for _ in range(int(self.elapsed_time)):
                    self.motor_simulation(target_pos)  # 根据目标位置进行电机模拟计算当前位置

                    self.weight_simulation(self.current_opening, self.material_coefficient)  # 根据当前位置计算当前重量
                    # time.sleep(0.003)
                    weights.append(self.current_weight)
                    openings.append(self.current_opening)
                    speeds.append(self.current_speed)
                    # print(
                    #     f"Current speed: {self.current_speed:.2f}, opening: {self.current_opening:.2f}, weight: {self.current_weight:.2f}, elapsed time: {self.elapsed_time} ms")
            self.client.write_registers(2, [int(self.current_weight)])  # 写入重量
            # print("write_weight",int(self.current_weight))
        # 开启允卸信号
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        new_value = current_value | (1 << 2)
        # 将新的值写回寄存器
        self.client.write_register(70, new_value)

        end_time = time.time()
        total_time = end_time - start_time
        # print("total_time:", total_time)
        final_weight = self.current_weight
        # print("current_weight:", final_weight)
        array = np.array([speeds, openings, weights])

        time.sleep(1)
        self.client.write_register(2, 0)  # 清空重量
        self.current_weight = 0

        return speeds, openings, weights, total_time, final_weight

    # def step(self, action):
    #     """执行动作，返回新的状态、奖励、是否结束"""
    #     # 限制 action 在合理范围内
    #     for key in action:
    #         action[key] = np.clip(action[key], *self.bounds[key])
    #
    #     # 计算称重误差（目标值 100000g，误差服从正态分布）
    #     actual_weight = (action["fast_weight"] * 2 + action["medium_weight"] + action["slow_weight"]) * 0.01
    #     weight_error = abs(actual_weight - self.target_weight)
    #
    #     # 计算加料时间（假设时间与加料速率成反比）
    #     feeding_time = 100000 / (action["fast_weight"] + action["medium_weight"] + action["slow_weight"]) * 10
    #
    #     # 计算奖励（目标是最小误差和最短时间）
    #     reward = - (weight_error + feeding_time)
    #
    #     # 更新状态
    #     self.state.update(action)
    #     new_state = {
    #         "weight_error": weight_error,
    #         "feeding_time": feeding_time
    #     }
    #
    #     return new_state, reward, False, {}

    def _run_simulation_offline(self, action):
        action["target_weight"] = self.target_weight
        self.controller.reset()
        self.controller.load_params(action)
        return self.controller.simulate_run()

    def step(self, action):
        """执行一步称重，返回新状态、奖励、是否完成"""

        # ✅ 动作格式检查与转换
        if isinstance(action, np.ndarray):
            action = {
                "fast_weight": action[0],
                "medium_weight": action[1],
                "slow_weight": action[2],
                "fast_opening": action[3],
                "medium_opening": action[4],
                "slow_opening": action[5],
                "fast_delay": action[6],
                "medium_delay": action[7],
                "slow_delay": action[8],
                "unload_delay": action[9],
            }

        # ✅ 动作范围裁剪
        for key in action:
            lower_bound, upper_bound = self.bounds[key]
            action[key] = np.clip(action[key], lower_bound, upper_bound)

        # ✅ 执行仿真
        if self.use_offline_sim:
            _, _, _, total_time, final_weight = self._run_simulation_offline(action)
        else:
            _, _, _, total_time, final_weight = self._run_simulation(action)

        # ✅ 计算误差
        weight_error = abs(final_weight - self.target_weight)
        time_error = abs(total_time - self.target_time)

        # ✅ 梯度误差奖励机制
        if weight_error <= 10:
            error_reward = +5000
        elif weight_error <= self.target_weight_err:
            error_reward = +4000
        elif weight_error <= self.target_weight_err*2:
            error_reward = +3000
        elif weight_error <= self.target_weight_err*4:
            error_reward = +1500
        elif weight_error <= self.target_weight_err*6:
            error_reward = -1000
        elif weight_error <= self.target_weight_err*8:
            error_reward = -2000
        elif weight_error <= self.target_weight_err*12:
            error_reward = -3000
        else:
            error_reward = -300 - (weight_error - 300) * 0.5

        # ✅ 时间误差惩罚
        if time_error <= 0.3 * self.target_time:
            time_reward = +500
        elif time_error <= 0.5 * self.target_time:
            time_reward = +300
        elif time_error <= 0.8 * self.target_time:
            time_reward = +100
        elif time_error <= 1.1 * self.target_time:
            time_reward = 10
        elif time_error <= 2 * self.target_time:
            time_reward = -100
        elif time_error <= 3 * self.target_time:
            time_reward = -200
        else:
            time_reward = -300 - (time_error - 3 * self.target_time) * 200

        # # ✅ 连续误差奖励（指数衰减）
        # error_reward = 6000 * np.exp(-weight_error / self.target_weight_err) - 500

        # # ✅ 时间惩罚（指数惩罚 + 平滑）
        # time_factor = time_error / self.target_time
        # time_reward = 3000 * np.exp(-time_factor) - 500  # 最低为 -300
        target_reward = 0
        if weight_error <= self.target_weight_err:
            time_factor = np.clip((3 * self.target_time - total_time) / self.target_time, 0, 3)
            target_reward = 2000 * np.tanh(time_factor)

        # # ✅ 达标目标范围内再加奖励（仅误差较小时）
        # target_reward = 0
        # if weight_error <= 25:
        #     time_factor = np.clip((3 * self.target_time - total_time) / self.target_time, 0, 3)
        #     target_reward = 2000 * np.tanh(time_factor)  # 平滑上升

        # ✅ 动作多样性奖励
        action_array = np.array(list(action.values()), dtype=np.float32)
        action_std = np.std(action_array)
        diversity_bonus = 0.05 * action_std

        # ✅ 控制边界惩罚（保持不变）
        boundary_penalty = 0
        for key, value in action.items():
            if key not in self.bounds:
                continue
            lower, upper = self.bounds[key]
            if value <= lower + 1e-3 or value >= upper - 1e-3:
                boundary_penalty -= 300

        # ✅ 汇总
        base_reward = -1000
        reward = base_reward + error_reward + time_reward + target_reward + diversity_bonus + boundary_penalty
        reward = max(reward, -3000)
        reward = min(reward, 6000)

        # ✅ 新状态向量（推荐使用归一化）
        new_state_vector = np.array([
            weight_error / self.target_weight_err,
            time_error / self.target_time
        ], dtype=np.float32)

        # ✅ 记录误差并判断是否结束
        self.recent_errors.append(weight_error)
        min_steps_required = self.max_steps // 2
        done = (
                self.step_counter >= self.max_steps or
                (self.step_counter >= min_steps_required and
                 len(self.recent_errors) == self.recent_errors.maxlen and
                 np.mean(self.recent_errors) < 25)
        )

        # ✅ 显示训练状态
        print(f"[Ep{self.episode_counter - 1} | Step{self.step_counter}] "
              f"Weight: {final_weight:.1f}g | Error: {weight_error:.1f}g | "
              f"Time: {total_time:.3f}s | Reward: {reward:.2f}")
        print(f"Action: {action}")

        # if done:
        #     print(f"✔️ Episode {self.episode_counter} finished. "
        #           f"Avg error (last 25): {np.mean(self.recent_errors):.2f}g")

        # # ✅ 写入日志
        # with open("weight_log.csv", "a") as f:
        #     f.write(f"{self.episode_counter},{self.step_counter},{final_weight:.1f},"
        #             f"{weight_error:.1f},{feeding_time:.3f},{reward:.2f}\n")

        # ✅ 步数更新
        self.step_counter += 1

        return new_state_vector, reward, done, {
            "final_weight": final_weight,
            "weight_error": weight_error,
            "total_time": total_time,
            "action": action
        }

class MultiConditionWeightEnv(WeightEnv):
    def __init__(self, config=None):
        super().__init__()

        if config is not None:
            self.target_weight = config.get("target_weight", self.target_weight)
            self.target_weight_err = config.get("target_weight_err", self.target_weight_err)
            self.target_time = config.get("target_time", self.target_time)

        if config and "bounds" in config:
            self.bounds = config["bounds"]

        # 确保更新归一化参数
        self._update_scaling_params()

        # 更新 action_dim
        self.action_dim = len(self.bounds)
