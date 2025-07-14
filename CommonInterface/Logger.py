import logging
import os

def init_logger(log_prefix=None, phase="train", subdir="logs", fallback_default=True):
    """
    初始化日志记录器
    :param log_prefix: 日志文件前缀，例如 "exp4_variant_25kg"
    :param phase: 训练阶段，如 "train" / "test"
    :param subdir: 日志保存目录
    :param fallback_default: 若未传 prefix 是否返回默认 logger（适配实验一）
    :return: logger 对象
    """
    if not log_prefix:
        if fallback_default:
            return logging.getLogger()
        else:
            return None

    os.makedirs(subdir, exist_ok=True)
    log_path = os.path.join(subdir, f"{phase}_{log_prefix}.log")

    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger