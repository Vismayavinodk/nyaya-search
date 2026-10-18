"""
Structured logging module for console and file-based runtime monitoring.
"""

import os
import sys
import logging
from typing import Optional


def get_logger(name: str = "IR_Legal_Retrieval", log_dir: Optional[str] = "logs") -> logging.Logger:
    """
    Initializes and returns a configured logger instance with standard formatters.

    Args:
        name: Name of the logger module.
        log_dir: Directory where log files are written. If None, file logging is skipped.

    Returns:
        logging.Logger configured for console stdout and file tracking.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid adding duplicate handlers if logger was already initialized
    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )

        # Console Stream Handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        # File Handler (if log_dir provided)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
            log_path = os.path.join(log_dir, "run.log")
            file_handler = logging.FileHandler(log_path, encoding="utf-8")
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

    return logger
