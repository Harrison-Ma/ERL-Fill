import os
import sys
import logging


class CallbackHandler(logging.Handler):
    """
    A custom logging handler that sends log messages to a callback function.

    This is especially useful in GUI applications or Jupyter notebooks where log
    messages need to be redirected to a text box or interactive display component.

    Args:
        callback (function): A callable that takes a single string argument.
                             Typically a function like `append_log(text)` in a GUI.
    """

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        """
        Emit a log record by formatting it and sending to the callback.

        Args:
            record (LogRecord): The log record to emit.
        """
        try:
            msg = self.format(record)
            self.callback(msg)
        except Exception:
            self.handleError(record)


def init_logger(log_prefix=None,
                phase="train",
                subdir="logs",
                fallback_default=True,
                to_console=True,
                log_callback=None):
    """
    Initialize and configure a flexible logger for file, console, and GUI output.

    This function sets up a logger with the ability to log to:
    - A specified log file (automatically created)
    - The system console (stdout)
    - A callback function for custom UIs (e.g., log window in GUI/Jupyter)

    Args:
        log_prefix (str): Prefix for the log filename (e.g., "exp1"). Required unless fallback_default is True.
        phase (str): Phase identifier to append in the filename (e.g., "train", "test").
        subdir (str): Directory to store log files. Defaults to "logs".
        fallback_default (bool): If True, returns the root logger when log_prefix is not provided.
        to_console (bool): If True, log messages will also be printed to stdout.
        log_callback (function): Optional callback to send formatted log strings to a custom handler (e.g., GUI).

    Returns:
        logging.Logger: Configured logger instance.
    """
    if not log_prefix:
        if fallback_default:
            logger = logging.getLogger()
            logger.setLevel(logging.INFO)
            return logger
        else:
            return None

    # ✅ Ensure the log directory exists
    os.makedirs(subdir, exist_ok=True)
    log_path = os.path.join(subdir, f"{log_prefix}_{phase}.log")

    # Get or create logger with unique name
    logger = logging.getLogger(log_prefix)
    logger.setLevel(logging.INFO)

    # ❌ Prevent duplicate handlers (especially on repeated calls)
    if logger.hasHandlers():
        logger.handlers.clear()

    # ✅ Standard log formatter
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

    # ➤ File handler: writes logs to a file
    file_handler = logging.FileHandler(log_path, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    # ➤ Console handler: prints logs to stdout
    if to_console:
        console_handler = logging.StreamHandler(stream=sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # ➤ Callback handler: sends logs to a callback (e.g., GUI display)
    if log_callback:
        callback_handler = CallbackHandler(log_callback)
        callback_handler.setFormatter(formatter)
        logger.addHandler(callback_handler)

    return logger
