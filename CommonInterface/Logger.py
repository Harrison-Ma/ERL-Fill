import logging
import os
import sys

class CallbackHandler(logging.Handler):
    """用于日志框（如 GUI/Jupyter 控件）输出"""
    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)

def init_logger(log_prefix=None, phase="train", subdir="logs",
                fallback_default=True, to_console=True, log_callback=None):
    """
    通用日志初始化器，兼容 Windows/Linux，支持控制台、文件、日志框多重输出
    :param log_prefix: 日志文件前缀，如 "exp1"
    :param phase: 日志阶段名，如 "train"/"test"
    :param subdir: 日志保存路径
    :param fallback_default: 未提供 prefix 时是否使用默认 logger
    :param to_console: 是否输出到控制台
    :param log_callback: GUI/Jupyter 等日志框回调函数 callback(text)
    :return: logger 对象
    """
    if not log_prefix:
        if fallback_default:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            return logger
        else:
            return None

    os.makedirs(subdir, exist_ok=True)
    log_path = os.path.join(subdir, f"{log_prefix}_{phase}.log")
    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)

    # 防止重复添加 handler（尤其多次调用时）
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # 写入文件
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # 控制台输出（跨平台兼容）
    if to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # 可选日志框输出
    if log_callback:
        callback_handler = CallbackHandler(log_callback)
        callback_handler.setFormatter(formatter)
        logger.addHandler(callback_handler)

    return logger
