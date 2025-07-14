import time
import random
import numpy as np

class VirtualWeightController:
    def __init__(self):
        self.reset()

    def reset(self):
        self.curr_weight = 0
        self.current_opening = 0
        self.current_speed = 0
        self.previous_time = time.time()
        self.params = {}
        self.registers = [0] * 300

    def write_register(self, address, value):
        self.registers[address] = value

    def write_registers(self, address, values):
        for i, val in enumerate(values):
            self.registers[address + i] = val

    def read_holding_registers(self, address, count=1):
        class Result:
            def __init__(self, values):
                self.registers = values
        return Result(self.registers[address:address + count])

    def load_params(self, action):
        self.params = {
            "target_weight": int(action["target_weight"]),
            "fast_weight": int(action["fast_weight"]),
            "medium_weight": int(action["medium_weight"]),
            "slow_weight": int(action["slow_weight"]),
            "fast_delay": int(action["fast_delay"]),
            "medium_delay": int(action["medium_delay"]),
            "slow_delay": int(action["slow_delay"]),
            "fast_pos": int(action["fast_opening"]),
            "medium_pos": int(action["medium_opening"]),
            "slow_pos": int(action["slow_opening"]),
            "unload_delay": int(action["unload_delay"]),
        }

    def motor_simulation(self, target_pos):
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
        random_number = random.uniform(-1, 1)
        self.curr_weight += opening * (1.0 + random_number)
        if self.curr_weight < 0:
            self.curr_weight = 0

    def simulate_run(self):
        speeds, openings, weights = [], [], []
        p = self.params

        system_delay_time = random.uniform(15, 50) #ms

        fast_pos = p["fast_pos"]
        medium_pos = p["medium_pos"]
        slow_pos = p["slow_pos"]

        fast_weight = p["fast_weight"]
        medium_weight = p["medium_weight"]
        slow_weight = p["slow_weight"]

        fast_delay = p["fast_delay"]
        medium_delay = p["medium_delay"]
        slow_delay = p["slow_delay"]
        unload_delay = p["unload_delay"]

        target_pos = fast_pos
        delay_finish = False

        max_duration = 10.0
        max_iterations = 5000
        no_change_limit = 500
        iteration = 0
        no_change_count = 0
        prev_weight = self.curr_weight

        use_real_time = 0
        start_time = time.time()
        self.previous_time = time.time()

        finish = False
        # print("going into weight circle......")

        tick = 0

        while not finish:
            self.current_time = time.time()
            self.elapsed_time = self.current_time - self.previous_time
            self.previous_time = self.current_time
            elapsed_ms = max(1, round(self.elapsed_time * 1000))  # 防止 range(0)

            if use_real_time:
                time.sleep(0.01)  # 降低 CPU 占用
            else:
                tick = tick + 1
            for _ in range(elapsed_ms):
                self.motor_simulation(target_pos)
                self.weight_simulation(self.current_opening)

                weights.append(self.curr_weight)
                openings.append(self.current_opening)
                speeds.append(self.current_speed)

                if self.curr_weight < fast_weight:
                    target_pos = fast_pos
                    if not delay_finish:
                        if use_real_time:
                            time.sleep(fast_delay / 1000.0)
                        else:
                            tick = tick + fast_delay
                        delay_finish = True
                        self.curr_weight = + target_pos * system_delay_time
                elif self.curr_weight < medium_weight:
                    target_pos = medium_pos
                    if not delay_finish:
                        if use_real_time:
                            time.sleep(medium_delay / 1000.0)
                        else:
                            tick = tick + medium_delay
                        delay_finish = True
                        self.curr_weight = + target_pos * system_delay_time
                elif self.curr_weight < slow_weight:
                    target_pos = slow_pos
                    if not delay_finish:
                        if use_real_time:
                            time.sleep(slow_delay / 1000.0)
                        else:
                            tick = tick + slow_delay
                        delay_finish = True
                        self.curr_weight = + target_pos * system_delay_time
                else:
                    finish = True
                    break
                iteration += 1
                if abs(self.curr_weight - prev_weight) < 0.01:
                    no_change_count += 1
                else:
                    no_change_count = 0
                prev_weight = self.curr_weight

                if iteration > max_iterations:
                    print("[警告] 超过最大迭代次数，强制退出防止死循环。")
                    finish = True
                    break
                if no_change_count > no_change_limit:
                    print("[警告] 权重长时间无变化，可能卡死，强制退出。")
                    finish = True
                    break
                if time.time() - start_time > max_duration:
                    print("[警告] 超过最大运行时间，强制退出防止死循环。")
                    finish = True
                    break
        if use_real_time:
            time.sleep(unload_delay / 1000.0)
            total_time = time.time() - start_time
        else:
            total_time = (tick + unload_delay) / 1000

        # print("total_time(tick): ", total_time)
        # print("total_time: ", time.time() - start_time)
        final_weight = self.curr_weight
        array = np.array([speeds, openings, weights])

        # time.sleep(1)
        self.curr_weight = 0

        return speeds, openings, weights, total_time, final_weight