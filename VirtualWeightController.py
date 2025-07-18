import time
import random


class VirtualWeightController:
    """
    Virtual simulation controller for a dynamic weight-based motor control system.
    Simulates motor movement, weight accumulation, and dynamic response logic.

    Intended for use in control experiments or reinforcement learning environments.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        """
        Resets the simulation to initial conditions.
        """
        self.curr_weight = 0
        self.current_opening = 0
        self.current_speed = 0
        self.previous_time = time.time()
        self.params = {}
        self.registers = [0] * 300  # Simulated Modbus registers

    def write_register(self, address, value):
        """
        Writes a value to a simulated Modbus register.

        Args:
            address (int): Register index
            value (int): Value to write
        """
        self.registers[address] = value

    def write_registers(self, address, values):
        """
        Writes multiple values to consecutive simulated registers.

        Args:
            address (int): Start register index
            values (list[int]): List of values to write
        """
        for i, val in enumerate(values):
            self.registers[address + i] = val

    def read_holding_registers(self, address, count=1):
        """
        Simulates Modbus read from holding registers.

        Args:
            address (int): Start register
            count (int): Number of registers to read

        Returns:
            object: Contains `.registers` list with requested values
        """
        class Result:
            def __init__(self, values):
                self.registers = values
        return Result(self.registers[address:address + count])

    def load_params(self, action):
        """
        Loads control parameters from a dictionary (usually an RL action).

        Args:
            action (dict): Contains target weights, delays, and opening positions
        """
        self.params = {
            "target_weight": int(action["target_weight"]),
            "fast_weight": int(action["fast_weight"]),
            "medium_weight": int(action["medium_weight"]),
            "slow_weight": int(action["slow_weight"]),
            "fast_delay": int(action["fast_delay"]),
            "medium_delay": int(action["medium_delay"]),
            "slow_delay": int(action["slow_delay"]),
            "fast_opening": int(action["fast_opening"]),
            "medium_opening": int(action["medium_opening"]),
            "slow_opening": int(action["slow_opening"]),
            "unload_delay": int(action["unload_delay"]),
        }

    def motor_simulation(self, target_pos):
        """
        Simulates motor movement with simple acceleration/deceleration physics.

        Args:
            target_pos (float): Desired opening position
        """
        decelerate_time = self.current_speed / 100000 if self.current_speed > 0 else 0
        decelerate_distance = 0.5 * 100000 * decelerate_time ** 2

        if abs(self.current_opening - target_pos) > 1:
            if self.current_opening < target_pos:
                if self.current_opening < target_pos - decelerate_distance:
                    self.current_speed += 5000000 / 1000
                else:
                    self.current_speed -= 100000 / 1000
                self.motor_direction = 1
            else:
                if self.current_opening > decelerate_distance:
                    self.current_speed += 5000000 / 1000
                else:
                    self.current_speed -= 100000 / 1000
                self.motor_direction = -1
        else:
            self.current_speed = 0
            self.current_opening = target_pos

        self.speed = self.current_speed * self.motor_direction
        self.current_opening += self.speed / 1000

    def weight_simulation(self, opening):
        """
        Simulates weight accumulation based on opening position and noise.

        Args:
            opening (float): Current opening position
        """
        random_number = random.uniform(-1, 1)
        self.curr_weight += opening * (1.0 + random_number)
        if self.curr_weight < 0:
            self.curr_weight = 0

    def simulate_run(self):
        """
        Runs the full simulation based on loaded parameters.

        Returns:
            tuple: (speeds, openings, weights, total_time_sec, final_weight)
        """
        speeds, openings, weights = [], [], []
        p = self.params

        system_delay_time = random.uniform(15, 50)  # Simulated system delay (ms)

        # Extract control thresholds and settings
        fast_pos = p["fast_opening"]
        medium_pos = p["medium_opening"]
        slow_pos = p["slow_opening"]

        fast_weight = p["fast_weight"]
        medium_weight = p["medium_weight"]
        slow_weight = p["slow_weight"]

        fast_delay = p["fast_delay"]
        medium_delay = p["medium_delay"]
        slow_delay = p["slow_delay"]
        unload_delay = p["unload_delay"]

        target_pos = fast_pos
        delay_finish = False

        # Safety settings
        max_duration = 10.0           # Max total simulation time (sec)
        max_iterations = 5000         # Max loop iterations
        no_change_limit = 500         # Max stable iterations before abort

        iteration = 0
        no_change_count = 0
        prev_weight = self.curr_weight

        use_real_time = 0  # Toggle between simulated or real-time delay
        start_time = time.time()
        self.previous_time = time.time()

        finish = False
        tick = 0  # Simulated time ticks

        while not finish:
            self.current_time = time.time()
            self.elapsed_time = self.current_time - self.previous_time
            self.previous_time = self.current_time
            elapsed_ms = max(1, round(self.elapsed_time * 1000))

            if use_real_time:
                time.sleep(0.01)
            else:
                tick += 1

            for _ in range(elapsed_ms):
                self.motor_simulation(target_pos)
                self.weight_simulation(self.current_opening)

                weights.append(self.curr_weight)
                openings.append(self.current_opening)
                speeds.append(self.current_speed)

                # Update target position based on weight thresholds
                if self.curr_weight < fast_weight:
                    target_pos = fast_pos
                    if not delay_finish:
                        tick += fast_delay if not use_real_time else time.sleep(fast_delay / 1000)
                        delay_finish = True
                        self.curr_weight += target_pos * system_delay_time
                elif self.curr_weight < medium_weight:
                    target_pos = medium_pos
                    if not delay_finish:
                        tick += medium_delay if not use_real_time else time.sleep(medium_delay / 1000)
                        delay_finish = True
                        self.curr_weight += target_pos * system_delay_time
                elif self.curr_weight < slow_weight:
                    target_pos = slow_pos
                    if not delay_finish:
                        tick += slow_delay if not use_real_time else time.sleep(slow_delay / 1000)
                        delay_finish = True
                        self.curr_weight += target_pos * system_delay_time
                else:
                    finish = True
                    break

                # Check stopping conditions
                iteration += 1
                if abs(self.curr_weight - prev_weight) < 0.01:
                    no_change_count += 1
                else:
                    no_change_count = 0
                prev_weight = self.curr_weight

                if iteration > max_iterations:
                    print("[Warning] Exceeded max iterations — forced exit.")
                    finish = True
                    break
                if no_change_count > no_change_limit:
                    print("[Warning] Weight stagnant — possible stall, exiting.")
                    finish = True
                    break
                if time.time() - start_time > max_duration:
                    print("[Warning] Simulation timed out — exiting.")
                    finish = True
                    break

        if use_real_time:
            time.sleep(unload_delay / 1000.0)
            total_time = time.time() - start_time
        else:
            total_time = (tick + unload_delay) / 1000.0

        final_weight = self.curr_weight
        self.curr_weight = 0  # Reset for next run

        return speeds, openings, weights, total_time, final_weight
