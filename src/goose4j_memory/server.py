from fastmcp import FastMCP
from .config import settings
from .logging_utils import configure_logger

logger = configure_logger()
logger.info("Configuration loaded.")
logger.info("Neo4j database: %s", settings.neo4j_database)
logger.info("Ollama base URL: %s", settings.ollama_base_url)
logger.info("Embedding model: %s", settings.ollama_embed_model)

mcp = FastMCP("neo4j-memory")

# Import tools after mcp exists so decorators register cleanly
from . import tools  # noqa: F401,E402


def main() -> None:
    mcp.run()