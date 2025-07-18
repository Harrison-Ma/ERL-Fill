import platform
import logging
import os
import numpy as np
import random
import time
from pymodbus.client import ModbusSerialClient
from CommonInterface.modbus_slave import modbus_slave_client
from VirtualWeightController import VirtualWeightController
from collections import deque

from CommonInterface.Logger import init_logger

# === Detect platform and set default serial port name ===
if platform.system() == 'Windows':
    port_name = 'COM4'  # Default COM port for Windows
else:
    port_name = '/dev/ttyUSB0'  # Default USB serial port for Unix/Linux

# Environment definition (fixed), default target is 25kg
class WeightEnv:
    def __init__(self):
        self.controller = VirtualWeightController()  # Simulated controller

        self.episode_counter = 0   # Current episode number
        self.step_counter = 0      # Step counter within the current episode
        self.max_steps = 100       # Max steps per episode

        self.recent_errors = deque(maxlen=int(self.max_steps / 5))  # Stores recent errors for smoothing

        # Action parameter bounds (min, max) for normalization
        self.bounds = {
            "fast_weight": (18000, 23000),      # Fast filling rate range (g/s)
            "medium_weight": (0, 1),            # Medium filling threshold range (placeholder)
            "slow_weight": (24900, 25025),      # Slow filling threshold range (g/s)

            "fast_opening": (35, 60),           # Opening percentage for fast fill
            "medium_opening": (3, 5),           # Opening percentage for medium fill
            "slow_opening": (3, 7),             # Opening percentage for slow fill

            "fast_delay": (100, 300),           # Delay before switching from fast to medium (ms)
            "medium_delay": (100, 200),         # Delay before switching from medium to slow (ms)
            "slow_delay": (100, 200),           # Delay before stopping at slow threshold (ms)

            "unload_delay": (300, 500)          # Delay for unloading material (ms)
        }

        # Precompute scale centers and ranges for normalization
        self.scale_centers = {}
        self.scale_ranges = {}
        for key, (low, high) in self.bounds.items():
            self.scale_centers[key] = (high + low) / 2
            self.scale_ranges[key] = (high - low) / 2

        self.state_dim = 2               # Environment state dimension (e.g., error, time)
        self.action_dim = len(self.bounds)  # Number of control parameters

        self.state = None                # Placeholder for current environment state
        self.agent = None                # Agent interface (for memory access)
        self.use_offline_sim = 1         # 1: use offline simulation mode, 0: real hardware

        self.target_weight = 25000       # Target weight in grams
        self.target_weight_err = 25      # Acceptable error threshold (grams)
        self.target_time = 2.5           # Ideal fill time (seconds)

        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.material_coefficient = 1.0  # Material coefficient (placeholder)

        self.registers = [0] * 300       # Simulated Modbus register array

        # Motor simulation parameters
        self.current_speed = 0           # Current filling speed
        self.motor_speed = 10000
        self.motor_acceleration = 5000000
        self.motor_deceleration = 100000
        self.motor_direction = 0
        self.speed = 0

        # Modbus client setup (used in online/hardware mode)
        self.client = ModbusSerialClient(
            port=port_name,
            baudrate=460800,
            parity='E',
            timeout=3
        )

        if self.use_offline_sim == 0:
            # Try to connect to real hardware if not using simulation
            if self.client.connect():
                print("Successfully connected to Modbus device")
            else:
                print("Failed to connect")

    def normalize_action(self, action):
        """
        Normalize neural network outputs (in [-1, 1]) to actual action value ranges.
        Returns a dictionary matching self.bounds.
        """
        scaled_action = {}
        for idx, key in enumerate(self.bounds):
            scaled_action[key] = action[idx] * self.scale_ranges[key] + self.scale_centers[key]
        return scaled_action

    def attach_agent(self, agent):
        """
        Attach a learning agent instance (optional).
        Enables access to memory for experience-based resets.
        """
        self.agent = agent

    def reset(self):
        """
        Reset environment and return the initial normalized state vector.

        Returns:
            state_vector (np.array): Normalized state [error_ratio, time_ratio]
        """
        ref_weight = 25000.0  # Reference values for normalization
        ref_time = 10.0

        if self.episode_counter == 0 or self.agent is None or len(self.agent.memory) < 10:
            # Initial episode or not enough experience → random state
            raw_state = np.array([
                random.uniform(0, self.target_weight_err),
                random.uniform(0, self.target_time)
            ], dtype=np.float32)
        else:
            # Sample past experiences and average for smoother reset
            samples = random.sample(self.agent.memory.buffer, 10)
            raw_state = np.mean([s[0] for s in samples], axis=0).astype(np.float32)

        # Normalize state
        state_vector = np.array([
            raw_state[0] / self.target_weight_err,
            raw_state[1] / self.target_time
        ], dtype=np.float32)

        # Reset environment internals
        self.current_weight = 0
        self.current_speed = 0
        self.current_opening = 0
        self.step_counter = 0
        self.episode_counter += 1
        self.recent_errors.clear()

        return state_vector

    def _state_to_vector(self):
        """
        Convert current internal state (dict) to numpy vector.
        Only used if `self.state` is a dict.
        """
        return np.array([self.state[key] for key in self.param_order], dtype=np.float32)

    def write_register(self, address, value):
        """
        Simulate writing a value to a Modbus register.
        """
        self.registers[address] = value

    def write_registers(self, address, values):
        """
        Simulate writing multiple values to consecutive Modbus registers.
        """
        for i, val in enumerate(values):
            self.registers[address + i] = val

    def read_holding_registers(self, address, count):
        """
        Simulate reading multiple Modbus holding registers.
        """
        return self.registers[address:address + count]

    def motor_simulation(self, target_pos):
        """
        Simulates motor movement toward a target position.

        Args:
            target_pos (float): The desired opening position of the motor.

        Updates:
            - Motor direction
            - Speed (acceleration/deceleration logic)
            - Current opening position
        """
        decelerate_time = self.current_speed / self.motor_deceleration
        decelerate_distance = 0.5 * self.motor_deceleration * np.power(decelerate_time, 2)

        if abs(self.current_opening - target_pos) > 1:
            if self.current_opening < target_pos:
                # Forward direction
                self.motor_direction = 1
                if self.current_opening < target_pos - decelerate_distance:
                    # Acceleration phase
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                else:
                    # Deceleration phase
                    self.current_speed -= self.motor_deceleration / 1000
            elif self.current_opening > target_pos:
                # Reverse direction
                self.motor_direction = -1
                if self.current_opening > decelerate_distance:
                    # Acceleration phase
                    if self.current_speed < self.motor_speed:
                        self.current_speed += self.motor_acceleration / 1000
                else:
                    # Deceleration phase
                    self.current_speed -= self.motor_deceleration / 1000
        else:
            # Target reached
            self.current_speed = 0
            self.current_opening = target_pos

        self.speed = self.current_speed * self.motor_direction
        self.current_opening += self.speed / 1000  # Update position

    def weight_simulation(self, opening, material_coeff):
        """
        Simulates weight increase based on opening and material coefficient.

        Args:
            opening (float): The current opening of the valve.
            material_coeff (float): The material coefficient affecting flow rate.
        """
        random_number = random.uniform(-1, 1)
        self.current_weight = int(self.current_weight + opening * (material_coeff + random_number))

    def _get_current_opening(self):
        """
        Determines the current valve opening based on the weight stage.

        Returns:
            float: Opening value for the current weight stage (fast, medium, slow).
        """
        if self.current_weight < 50000:
            return self.state["fast_opening"]
        elif self.current_weight < 80000:
            return self.state["medium_opening"]
        else:
            return self.state["slow_opening"]

    def _run_simulation(self, action):
        """
        Main simulation loop for one episode. Executes the entire filling process.

        Args:
            action (dict): Dictionary of control parameters from the agent.

        Returns:
            tuple: (speeds, openings, weights, total_time, final_weight)
                - speeds: List of motor speeds over time
                - openings: List of valve openings over time
                - weights: List of weight readings over time
                - total_time: Duration of the entire process (seconds)
                - final_weight: Final weight achieved at the end
        """
        # Convert action dict to Modbus register format
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
        print("Arg:", Arg)
        self.client.write_registers(200, Arg)

        # Send start signal (bit 0 = 1)
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        self.client.write_register(70, current_value | (1 << 0))

        # Send allow-feed signal (bit 1 = 1)
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        self.client.write_register(70, current_value | (1 << 1))

        start_time = time.time()
        self.client.write_registers(5, 0)  # Reset status register

        speeds, openings, weights = [], [], []
        PackFinish = 0
        self.previous_time = time.time()

        while not PackFinish:
            self.current_time = time.time()
            self.elapsed_time = self.current_time - self.previous_time
            self.previous_time = self.current_time
            self.elapsed_time = round(self.elapsed_time * 1000)

            # Read 124 control registers (e.g., Modbus registers from controller)
            controler_modbus_reg = self.client.read_holding_registers(address=0, count=124)
            target_status = controler_modbus_reg.registers[6]
            PackFinish = (target_status & (1 << 5))  # Check if filling is done

            target_pos = controler_modbus_reg.registers[10]  # Read target motor position

            if self.elapsed_time < 200:
                for _ in range(int(self.elapsed_time)):
                    self.motor_simulation(target_pos)
                    self.weight_simulation(self.current_opening, self.material_coefficient)

                    weights.append(self.current_weight)
                    openings.append(self.current_opening)
                    speeds.append(self.current_speed)

            self.client.write_registers(2, [int(self.current_weight)])  # Update current weight to register

        # Send unload permission (bit 2 = 1)
        result = self.client.read_holding_registers(70)
        current_value = result.registers[0]
        self.client.write_register(70, current_value | (1 << 2))

        end_time = time.time()
        total_time = end_time - start_time
        final_weight = self.current_weight

        time.sleep(1)
        self.client.write_register(2, 0)  # Clear weight register
        self.current_weight = 0

        return speeds, openings, weights, total_time, final_weight

    def _run_simulation_offline(self, action):
        """
        Simulate one run in an offline environment.

        Args:
            action (dict): Control parameters.

        Returns:
            tuple: Simulation result from controller (includes final weight and time).
        """
        action["target_weight"] = self.target_weight
        self.controller.reset()
        self.controller.load_params(action)
        return self.controller.simulate_run()

    def _run_real(self, action):
        """
        Run the real hardware system with the given action.

        Args:
            action (dict): Control parameters.

        Returns:
            tuple: Real environment results including total time and final weight.
        """
        action["target_weight"] = self.target_weight
        Arg = [
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
        _, _, _, total_time, final_weight = modbus_slave_client._run_read_and_write(Arg)
        return _, _, _, total_time, final_weight

    def step(self, action):
        """
        Executes one step in the environment.

        Args:
            action (dict or ndarray): Control action either as a dict or ndarray.

        Returns:
            tuple: (state, reward, done, info)
                - state (ndarray): Normalized error-based state vector.
                - reward (float): Scalar reward signal.
                - done (bool): Whether the episode has finished.
                - info (dict): Additional debug information.
        """

        # ✅ Convert action from ndarray to dictionary if needed
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

        # ✅ Clip actions within predefined bounds
        for key in action:
            lower_bound, upper_bound = self.bounds[key]
            action[key] = np.clip(action[key], lower_bound, upper_bound)

        # ✅ Choose simulation mode: 0=simulator, 1=offline, 2=real hardware
        if self.use_offline_sim == 1:
            _, _, _, total_time, final_weight = self._run_simulation_offline(action)
        elif self.use_offline_sim == 0:
            _, _, _, total_time, final_weight = self._run_simulation(action)
        elif self.use_offline_sim == 2:
            _, _, _, total_time, final_weight = self._run_real(action)

        # ✅ Compute weight and time error
        weight_error = abs(final_weight - self.target_weight)
        feeding_time = total_time * 1000  # in milliseconds
        time_error = abs(total_time - self.target_time)

        # ✅ Reward shaping: weight error
        if weight_error <= 5:
            error_reward = 8000 - 200 * weight_error
        elif weight_error <= 10:
            error_reward = 6000 - 120 * (weight_error - 5)
        elif weight_error <= 15:
            error_reward = 4000 - 40 * (weight_error - 10)
        elif weight_error <= 20:
            error_reward = 2000 - 20 * (weight_error - 15)
        elif weight_error <= 25:
            error_reward = -1000 - 10 * (weight_error - 20)
        elif weight_error <= 50:
            error_reward = -3000 - 5 * (weight_error - 25)
        else:
            error_reward = -4000 - 0.5 * (weight_error - 50)

        # ✅ Reward shaping: time error
        if time_error <= 0.1 * self.target_time:
            time_reward = +3000
        elif time_error <= 0.3 * self.target_time:
            time_reward = +1000
        elif time_error <= 0.5 * self.target_time:
            time_reward = +500
        elif time_error <= 0.8 * self.target_time:
            time_reward = -1000
        elif time_error <= 1.2 * self.target_time:
            time_reward = -2000
        elif time_error <= 2 * self.target_time:
            time_reward = -3000
        else:
            time_reward = -3000 - (time_error - 2 * self.target_time) * 500

        # ✅ Optional smooth reward based on how fast it reaches target (tanh-shaped)
        target_reward = 0
        if weight_error <= 25:
            time_factor = np.clip((3 * self.target_time - total_time) / self.target_time, 0, 3)
            target_reward = 2000 * np.tanh(time_factor)

        # ✅ Boundary penalty to encourage actions away from min/max
        boundary_penalty = 0
        penalty_per_action = 300
        buffer_ratio = 0.05

        for key, value in action.items():
            if key not in self.bounds:
                continue
            lower, upper = self.bounds[key]
            buffer = (upper - lower) * buffer_ratio
            if value <= lower + buffer or value >= upper - buffer:
                boundary_penalty -= penalty_per_action

        print("boundary_penalty: ", boundary_penalty)

        # ✅ Total reward aggregation with limits
        base_reward = -5000
        reward = base_reward + error_reward + time_reward + target_reward + boundary_penalty
        reward = max(reward, -10000)
        reward = min(reward, 10000)

        # ✅ New state (normalized features)
        new_state_vector = np.array([
            weight_error / self.target_weight_err,
            time_error / self.target_time
        ], dtype=np.float32)

        # ✅ Determine if the episode is done
        self.recent_errors.append(weight_error)
        min_steps_required = self.max_steps // 2
        done = (
                self.step_counter >= self.max_steps or
                (self.step_counter >= min_steps_required and
                 len(self.recent_errors) == self.recent_errors.maxlen and
                 np.mean(self.recent_errors) < 25)
        )

        # ✅ Print step info
        print(f"[Ep{self.episode_counter - 1} | Step{self.step_counter}] "
              f"Weight: {final_weight:.1f}g | Error: {weight_error:.1f}g | "
              f"Time: {total_time:.3f}s | Reward: {reward:.2f}")
        print(f"Action: {action}")

        # ✅ Step counter increment
        self.step_counter += 1

        return new_state_vector, reward, done, {
            "final_weight": final_weight,
            "weight_error": weight_error,
            "total_time": total_time,
            "action": action
        }
