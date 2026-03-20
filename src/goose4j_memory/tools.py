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
    Search Neo4j memory for relevant prior facts and return a ranked list.

    Call this tool BEFORE answering any question that involves:
    - The user's name, role, preferences, or background
    - Prior decisions or project context
    - Anything the user may have mentioned in a past session

    Parameters:
    - query: A short, specific phrase describing what you are looking for.
      Use noun phrases, not questions. Good: "user's preferred editor".
      Bad: "what does the user like?".
    - memory_type: Filter to one memory type. Must be one of: "entity",
      "semantic", "task", "episodic". Omit (null) to search all types.
    - top_k: Number of results to return. Default 5 is suitable for most
      queries; increase only when broad context is needed.
    - expand_concepts: When true (default), widens the search using related
      concepts found in the initial results. Set to false only when you need
      a fast, narrow lookup and are confident the query is highly specific.

    Return value: A list of memory records, each with "text", "memory_type",
    and a relevance score. An empty list means no relevant memories exist —
    do not retry with the same query.

    Good examples:
    - query="user's name", memory_type="entity"
    - query="user collaboration preferences", memory_type="semantic"
    - query="active task goose memory adapter", memory_type="task"
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
    Store exactly one durable fact in Neo4j memory.

    Call this tool when you learn something worth remembering across sessions:
    a new fact about the user, a preference, a task milestone, or a stable
    project detail. Do NOT call it for greetings, narration, or information
    already present in memory.

    Parameters:
    - content: A single, short, factual statement in third-person. One fact
      per call. Do not combine multiple facts.
    - memory_type: Must be exactly one of: "entity", "semantic", "task",
      "episodic". Choose the best fit:
        - "entity"   → identity or stable attributes (name, role, location)
        - "semantic" → preferences, expertise, durable facts about the world
        - "task"     → active work state or concrete next steps
        - "episodic" → specific past events tied to a session or time
    - conversation_id: Optional session identifier to group related memories.
      Use a consistent string (e.g., date + topic) within a session.
    - concepts: Optional list of short keyword tags (e.g., ["python", "neo4j"]).
      If omitted, concepts are extracted automatically.

    Return value: A dict with "status" ("stored" or "rejected") and "stored"
    (true/false). If rejected with reason "duplicate", the fact already exists
    — do not retry. If rejected with reason "quality_rejected", the content
    was too vague or short — rewrite it with more specific detail before
    retrying.

    Good examples:
    - content="User's name is Richard.", memory_type="entity"
    - content="User has a Ph.D. in Industrial/Organizational Psychology.", memory_type="entity"
    - content="User prefers a thought partner who challenges reasoning and assumptions.", memory_type="semantic"

    Bad examples:
    - content="Hello Richard, nice to meet you...", memory_type="semantic"
    - content="User is named Richard and prefers a thought partner and has a Ph.D...", memory_type="semantic"
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
    Return a health summary for the Neo4j memory adapter.

    Only call this when the user explicitly asks about memory system health,
    connection status, or memory counts. Do not call during normal conversation.
    """
    return adapter_status()


@mcp.tool()
def memory_setup() -> dict[str, Any]:
    """
    Create or update Neo4j schema, constraints, indexes, and vector index.

    Only call this when the user explicitly asks to set up or reset the memory
    schema, or when memory_status reports that the vector index is missing.
    Do not call during normal conversation. Safe to run multiple times.
    """
    return setup_schema()


@mcp.tool()
def memory_cleanup() -> dict[str, Any]:
    """
    Normalize older memory records and migrate legacy schema details.

    Only call this when the user explicitly asks to clean up or migrate memory
    data, or after upgrading from an older adapter version. Do not call during
    normal conversation.
    """
    return cleanup_legacy_schema()
