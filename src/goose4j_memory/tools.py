from typing import Any
from .server import mcp
from .memory_service import adapter_status, retrieve_memory, store_memory
from .schema import cleanup_legacy_schema, setup_schema
from .mtypes import MemoryType


@mcp.tool()
def memory_search(
    query: str,
    top_k: int = 5,
    expand_concepts: bool = True,
    memory_type: MemoryType | None = None,
) -> list[dict[str, Any]]:
    """
    Search the Neo4j memory graph for relevant prior context.

    Use this for recall, previous decisions, stable preferences, project state,
    or related concepts discussed earlier.
    """
    return retrieve_memory(
        query=query,
        top_k=top_k,
        expand_concepts=expand_concepts,
        memory_type=memory_type,
    )


@mcp.tool()
def memory_store(
    content: str,
    conversation_id: str | None = None,
    concepts: list[str] | None = None,
    memory_type: MemoryType = "semantic",
) -> dict[str, Any]:
    """
    Store a durable memory in Neo4j.

    Use this only for durable facts, stable preferences, decisions,
    or ongoing project context. Do not store transient chatter.
    """
    return store_memory(
        text=content,
        conversation_id=conversation_id,
        concepts=concepts,
        memory_type=memory_type,
    )


@mcp.tool()
def memory_status() -> dict[str, Any]:
    """
    Return a quick health summary for the Neo4j memory adapter.
    """
    return adapter_status()


@mcp.tool()
def memory_setup() -> dict[str, Any]:
    """
    Create or update the Neo4j schema, constraints, indexes, and vector index.

    Safe to run multiple times.
    """
    return setup_schema()


@mcp.tool()
def memory_cleanup() -> dict[str, Any]:
    """
    Normalize older memory records and migrate legacy schema details.

    Use this when upgrading from older adapter versions.
    """
    return cleanup_legacy_schema()