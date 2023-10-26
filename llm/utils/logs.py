import logging

from llm import settings


def get_logger(name: str = "saturn-llm") -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.hasHandlers():
        return logger

    logger.setLevel(settings.LOG_LEVEL)

    handler = logging.StreamHandler()
    handler.setLevel(settings.LOG_LEVEL)

    formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger
