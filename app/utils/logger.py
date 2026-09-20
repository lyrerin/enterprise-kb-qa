import logging
import sys
import os
from app.config import LOG_LEVEL
from logging.handlers import RotatingFileHandler

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

_FORMAT = '%(asctime)s | %(levelname)-7s | %(name)s | %(message)s'
_DATEFMT = '%Y-%m-%d %H:%M:%S'

def _build_logger():
    logger = logging.getLogger("enterprise-kb")
    if logger.handlers:
        return logger
    logger.setLevel(getattr(logging,LOG_LEVEL.upper(),logging.INFO))
    logger.propagate = False

    formatter = logging.Formatter(_FORMAT, datefmt=_DATEFMT)

    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)

    file_handler = RotatingFileHandler(
        os.path.join(LOG_DIR,'app.log'),
        maxBytes = 1_000_000, backupCount = 3,encoding='utf-8'
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

logger = _build_logger()
