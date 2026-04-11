import logging
import sys


def get_logger(name: str) -> logging.Logger:
    """Provides a pre-configured logger with a modern format.

    Args:
        name: Logger name (typically __name__).

    Returns:
        A logging.Logger instance.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        formatter = logging.Formatter(
            "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)

    return logger
