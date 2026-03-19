import logging
import os
import sys


def configure_logger() -> logging.Logger:
    logging.basicConfig(
        level=os.getenv("LOG_LEVEL", "INFO").upper(),
        stream=sys.stderr,
        format="[neo4j-memory] %(levelname)s %(message)s",
    )
    logger = logging.getLogger("neo4j-memory")
    logger.setLevel(getattr(logging, os.getenv("LOG_LEVEL", "INFO").upper(), logging.INFO))
    return logger