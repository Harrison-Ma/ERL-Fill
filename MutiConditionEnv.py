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

# === Experiment configuration table ===
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

# Environment definition (default is 25kg)
class WeightEnv:
    def __init__(self):
        self.target_weight = 25000
        self.target_weight_err = 25
        self.target_time = 2.5

        self.controller = VirtualWeightController()

        self.episode_counter = 0  # Current episode number
        self.step_counter = 0  # Current step in the episode

        self.max_steps = 100  # Maximum steps per episode

        self.recent_errors = deque(maxlen=int(self.max_steps/5))
        # Parameter ranges (lower and upper bounds)
        self.bounds = {
            "fast_weight": (7000, 18000),  # Fast feed weight range (g)
            "medium_weight": (0, 1),       # Medium feed weight range (g)
            "slow_weight": (24900, 25000), # Slow feed weight range (g)

            "fast_opening": (35, 75),      # Fast opening range (%)
            "medium_opening": (3, 5),      # Medium opening range (%)
            "slow_opening": (5, 20),       # Slow opening range (%)

            "fast_delay": (100, 300),      # Fast delay range (ms)
            "medium_delay": (100, 200),    # Medium delay range (ms)
            "slow_delay": (100, 200),      # Slow delay range (ms)

            "unload_delay": (300, 500)     # Unload delay range (ms)
        }
        self._update_scaling_params()

        self.scale_centers = {}
        self.scale_ranges = {}
        for key, (low, high) in self.bounds.items():
            self.scale_centers[key] = (high + low) / 2
            self.scale_ranges[key] = (high - low) / 2

        self.state_dim = 2 # Output of each weighing process, e.g. error, time, rate, etc.
        self.action_dim = len(self.bounds) # Input control variables for each weighing, e.g. fast/medium/slow targets, opening, delay, etc.

        # Initial state
        self.state = None

        self.agent = None  # Reserved agent interface

        self.use_offline_sim = 1

        self.target_weight = 25000  # Target weight 25kg
        self.target_weight_err = 25  # Target weight error 25g
        self.target_time = 2
        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.material_coefficient = 1.0  # Material coefficient, assumed to be 1.0
        self.registers = [0] * 300  # Simulated Modbus registers

        # Opening/sec
        self.current_speed = 0  # Current speed
        self.motor_speed = 10000  # Motor speed
        self.motor_acceleration = 5000000  # Motor acceleration
        self.motor_deceleration = 100000  # Motor deceleration
        self.motor_direction = 0  # Motor direction
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
        """Map network output [-1,1] actions to actual parameter ranges"""
        scaled_action = {}
        for idx, key in enumerate(self.bounds):
            scaled_action[key] = action[idx] * self.scale_ranges[key] + self.scale_centers[key]
        return scaled_action

    def attach_agent(self, agent):
        self.agent = agent  # Receive agent (for accessing its memory)

    def reset(self):
        """Reset environment and return normalized initial state vector"""

        # Initialize state vector
        if self.episode_counter == 0 or self.agent is None or len(self.agent.memory) < 10:
            # First round or insufficient experience → random initialization and normalization
            raw_state = np.array([
                random.uniform(0, self.target_weight_err),
                random.uniform(0, self.target_time)
            ], dtype=np.float32)
        else:
            # Otherwise → sample experience buffer and average (already normalized)
            samples = random.sample(self.agent.memory.buffer, 10)
            raw_state = np.mean([s[0] for s in samples], axis=0).astype(np.float32)

        # State normalization (skip if already normalized)
        state_vector = np.array([
            raw_state[0] / self.target_weight_err,
            raw_state[1] / self.target_time
        ], dtype=np.float32)

        # Reset environment state variables
        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.step_counter = 0
        self.episode_counter += 1
        self.recent_errors.clear()

        return state_vector

    def _state_to_vector(self):
        """Convert dict state to vector"""
        return np.array([self.state[key] for key in self.param_order], dtype=np.float32)

    def write_register(self, address, value):
        self.registers[address] = value

    def write_registers(self, address, values):
        for i, val in enumerate(values):
            self.registers[address + i] = val

    def read_holding_registers(self, address, count):
        return self.registers[address:address + count]

    # def motor_simulation(self, target_pos):
    #     # Simplified motor simulation: directly set speed to 1% of position difference
    #     self.current_speed = abs(target_pos - self.registers[10]) * 0.01

    def motor_simulation(self, target_pos):
        # Update every millisecond
        # print("target_pos ", target_pos)
        # print("self.current_opening ", self.current_opening)

        # Calculate deceleration time
        decelerate_time = self.current_speed / self.motor_deceleration
        # Calculate deceleration distance
        decelerate_distance = 0.5 * self.motor_deceleration * np.power(decelerate_time, 2)
        if abs(self.current_opening - target_pos) > 1:
            # Update speed
            if self.current_opening < target_pos:
                # Forward
                self.motor_direction = 1
                if self.current_opening < target_pos - decelerate_distance:
                    # Acceleration phase
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                else:
                    # Deceleration phase
                    self.current_speed -= self.motor_deceleration / 1000
            elif self.current_opening > target_pos:
                # Reverse
                self.motor_direction = -1
                if self.current_opening > decelerate_distance:
                    # Acceleration phase
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                else:
                    # Deceleration phase
                    self.current_speed -= self.motor_deceleration / 1000
        else:
            self.current_speed = 0
            self.current_opening = target_pos

        self.speed = self.current_speed * self.motor_direction
        self.current_opening += self.speed / 1000

    def weight_simulation(self, opening, material_coeff):
        random_number = random.uniform(-1, 1)
        # Simplified weight simulation: weight increase = opening * material coefficient * time interval
        # Assume time interval is 1ms (0.001s)
        self.current_weight = int(self.current_weight + opening * (material_coeff + random_number))

    # def weight_simulation(opening, coefficient):
    #     global current_weight
    #     random_number = random.uniform(-1, 1)
    #     # Update current weight, add random number
    #     current_weight = int(current_weight + opening * (coefficient + random_number))

    def _get_current_opening(self):
        # Return corresponding opening by stage (simplified logic)
        if self.current_weight < 50000:  # Fast feed stage
            return self.state["fast_opening"]
        elif self.current_weight < 80000:  # Medium feed stage
            return self.state["medium_opening"]
        else:  # Slow feed stage
            return self.state["slow_opening"]

    def _run_simulation(self, action):
        # Write control parameters to registers
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

        # Enable start signal
        # Read current register value (assume holding or input register)
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        # Set bit 0 to 1
        new_value = current_value | (1 << 0)
        # Write new value back to register
        self.client.write_register(70, new_value)

        # Enable feed signal
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        # Set bit 1 to 1
        new_value = current_value | (1 << 1)
        # Write new value back to register
        self.client.write_register(70, new_value)

        # Start timing
        start_time = time.time()
        self.client.write_registers(5, 0)  # Reset status

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
            controler_modbus_reg = self.client.read_holding_registers(address=0, count=124)  # Read controller board data
            target_status = controler_modbus_reg.registers[6]  # Get current running status of controller board
            PackFinish = (target_status & (1 << 5))  # Set value signal

            target_pos = controler_modbus_reg.registers[10]  # Get target position

            if self.elapsed_time < 200:
                for _ in range(int(self.elapsed_time)):
                    self.motor_simulation(target_pos)  # Simulate motor position based on target position
                    self.weight_simulation(self.current_opening, self.material_coefficient)  # Calculate current weight based on current position
                    weights.append(self.current_weight)
                    openings.append(self.current_opening)
                    speeds.append(self.current_speed)
            self.client.write_registers(2, [int(self.current_weight)])  # Write weight
        # Enable unload signal
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        new_value = current_value | (1 << 2)
        # Write new value back to register
        self.client.write_register(70, new_value)

        end_time = time.time()
        total_time = end_time - start_time
        final_weight = self.current_weight
        array = np.array([speeds, openings, weights])

        time.sleep(1)
        self.client.write_register(2, 0)  # Clear weight
        self.current_weight = 0

        return speeds, openings, weights, total_time, final_weight

    # def step(self, action):
    #     """Execute action, return new state, reward, done"""
    #     for key in action:
    #         action[key] = np.clip(action[key], *self.bounds[key])
    #     actual_weight = (action["fast_weight"] * 2 + action["medium_weight"] + action["slow_weight"]) * 0.01
    #     weight_error = abs(actual_weight - self.target_weight)
    #     feeding_time = 100000 / (action["fast_weight"] + action["medium_weight"] + action["slow_weight"]) * 10
    #     reward = - (weight_error + feeding_time)
    #     self.state.update(action)
    #     new_state = {
    #         "weight_error": weight_error,
    #         "feeding_time": feeding_time
    #     }
    #     return new_state, reward, False, {}

    def _run_simulation_offline(self, action):
        action["target_weight"] = self.target_weight
        self.controller.reset()
        self.controller.load_params(action)
        return self.controller.simulate_run()

    def step(self, action):
        """Execute one weighing step, return new state, reward, done"""

        # Action format check and conversion
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

        # Clip action to valid range
        for key in action:
            lower_bound, upper_bound = self.bounds[key]
            action[key] = np.clip(action[key], lower_bound, upper_bound)

        # Run simulation
        if self.use_offline_sim:
            _, _, _, total_time, final_weight = self._run_simulation_offline(action)
        else:
            _, _, _, total_time, final_weight = self._run_simulation(action)

        # Calculate errors
        weight_error = abs(final_weight - self.target_weight)
        time_error = abs(total_time - self.target_time)

        # Gradient error reward mechanism
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

        # Time error penalty
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

        target_reward = 0
        if weight_error <= self.target_weight_err:
            time_factor = np.clip((3 * self.target_time - total_time) / self.target_time, 0, 3)
            target_reward = 2000 * np.tanh(time_factor)

        # Action diversity bonus
        action_array = np.array(list(action.values()), dtype=np.float32)
        action_std = np.std(action_array)
        diversity_bonus = 0.05 * action_std

        # Control boundary penalty
        boundary_penalty = 0
        for key, value in action.items():
            if key not in self.bounds:
                continue
            lower, upper = self.bounds[key]
            if value <= lower + 1e-3 or value >= upper - 1e-3:
                boundary_penalty -= 300

        # Reward summary
        base_reward = -1000
        reward = base_reward + error_reward + time_reward + target_reward + diversity_bonus + boundary_penalty
        reward = max(reward, -3000)
        reward = min(reward, 6000)

        # New state vector (recommended normalized)
        new_state_vector = np.array([
            weight_error / self.target_weight_err,
            time_error / self.target_time
        ], dtype=np.float32)

        # Record error and check if done
        self.recent_errors.append(weight_error)
        min_steps_required = self.max_steps // 2
        done = (
                self.step_counter >= self.max_steps or
                (self.step_counter >= min_steps_required and
                 len(self.recent_errors) == self.recent_errors.maxlen and
                 np.mean(self.recent_errors) < 25)
        )

        # Display training status
        print(f"[Ep{self.episode_counter - 1} | Step{self.step_counter}] "
              f"Weight: {final_weight:.1f}g | Error: {weight_error:.1f}g | "
              f"Time: {total_time:.3f}s | Reward: {reward:.2f}")
        print(f"Action: {action}")

        # Step update
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

        # Ensure normalization parameters are updated
        self._update_scaling_params()

        # Update action_dim
        self.action_dim = len(self.bounds)
