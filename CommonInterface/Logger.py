import logging
import os


def init_logger(log_prefix=None, phase="train", subdir="logs", fallback_default=True):
    """
    Initialize a logger for saving experiment logs to a file.

    Args:
        log_prefix (str): Prefix for the log file name (e.g., "exp4_variant_25kg").
                          If None and fallback_default is True, returns the default logger.
        phase (str): Training phase, used to differentiate logs (e.g., "train", "test").
        subdir (str): Directory to store the log file. Will be created if it doesn't exist.
        fallback_default (bool): If True and log_prefix is not provided, returns default logger.

    Returns:
        logging.Logger: Configured logger instance, or None if prefix is not given and fallback is disabled.
    """
    if not log_prefix:
        if fallback_default:
            return logging.getLogger()  # Return the root logger
        else:
            return None  # Explicitly return None if no prefix and fallback is disabled

    os.makedirs(subdir, exist_ok=True)  # Ensure the log directory exists
    log_path = os.path.join(subdir, f"{phase}_{log_prefix}.log")  # Construct full log path

    logger = logging.getLogger(log_prefix)  # Create or get a named logger
    logger.setLevel(logging.INFO)  # Set logging level to INFO

    # Avoid adding multiple handlers if the logger already has them
    if not logger.handlers:
        file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")  # Overwrite each run
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")  # Log format
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger
