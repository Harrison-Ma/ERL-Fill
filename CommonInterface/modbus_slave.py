import ctypes
# slave.py

import time
from threading import Thread
from pymodbus.datastore import ModbusSlaveContext, ModbusSequentialDataBlock, ModbusServerContext
from pymodbus.server import StartTcpServer


class ModbusSlaveHandler:
    def __init__(self, ip="0.0.0.0", port=5020):
        self.store = ModbusSlaveContext(
            hr=ModbusSequentialDataBlock(0, [0]*30)  # 30个寄存器，留出空间
        )
        self.context = ModbusServerContext(slaves=self.store, single=True)

        # 独立线程启动服务器
        self.server_thread = Thread(target=self.start_server, args=(ip, port), daemon=True)
        self.server_thread.start()

    def start_server(self, ip, port):
        print(f"[从站] 启动 Modbus TCP Server 在 {ip}:{port}")
        StartTcpServer(context=self.context, address=(ip, port))

    def _run_read_and_write(self, action):
        if len(action) != 10:
            raise ValueError("action 列表必须是10个元素")

        # 清理旧状态
        self.context[0x00].setValues(3, 10, [0])  # 状态标志
        self.context[0x00].setValues(3, 17, [0])  # 结果标志

        # 写入参数
        self.context[0x00].setValues(3, 0, action)
        print(f"[从站] 参数已写入：{action}")
        self.context[0x00].setValues(3, 10, [1])  # 通知主站参数已写入
        print(f"[从站] 参数已写入，等待主站读取...")

        while True:
            status = self.context[0x00].getValues(3, 10, count=1)[0]
            if status == 2:
                print("[从站] 主站读取完成，状态清零")
                self.context[0x00].setValues(3, 10, [0])
                break
            time.sleep(0.05)

        print("[从站] 等待主站写入结果...")
        while True:
            flag = self.context[0x00].getValues(3, 19, count=1)[0]
            if flag == 1:
                result = self.context[0x00].getValues(3, 15, count=4)
                total_time = self.convert_to_signed_32bit(result[0], result[1])
                final_weight = self.convert_to_signed_32bit(result[2], result[3])
                self.context[0x00].setValues(3, 19, [0])  # 清除标志
                print(f"[从站] 收到结果：time={total_time}, weight={final_weight}")
                return None, None, None, total_time, final_weight
            time.sleep(0.05)

    def convert_to_signed_32bit(self, high, low):
        """ 将 2 个 16 位寄存器转换为 32 位有符号整数 """
        value_32bit = (low << 16) | high  # 先按照无符号整数处理
        signed_value = ctypes.c_int32(value_32bit).value  # 转换为有符号整数
        return signed_value


modbus_slave_client = ModbusSlaveHandler(ip="192.168.31.54", port=5020)

# 示例运行主逻辑
# if __name__ == "__main__":
#     slave = ModbusSlaveHandler(ip="192.168.31.54", port=5020)
#
#     for i in range(3):  # 多轮调用测试
#         action = [i + 1] * 10
#         print(f"🔁 第 {i+1} 次调用 _run_real")
#         _, _, _, total_time, final_weight = slave._run_real(action)
#         print(f"✅ 第 {i+1} 次返回: time={total_time}, weight={final_weight}")
#         time.sleep(1)
