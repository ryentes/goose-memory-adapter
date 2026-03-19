from .db import run_query
from .embeddings import infer_embedding_dimension

CONSTRAINTS = [
    """
    CREATE CONSTRAINT memory_id IF NOT EXISTS
    FOR (m:Memory)
    REQUIRE m.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT concept_name IF NOT EXISTS
    FOR (c:Concept)
    REQUIRE c.name IS UNIQUE
    """,
    """
    CREATE CONSTRAINT conversation_id IF NOT EXISTS
    FOR (c:Conversation)
    REQUIRE c.id IS UNIQUE
    """,
]

INDEXES = [
    """
    CREATE INDEX memory_source IF NOT EXISTS
    FOR (m:Memory)
    ON (m.source)
    """,
    """
    CREATE INDEX memory_quality_passed IF NOT EXISTS
    FOR (m:Memory)
    ON (m.quality_passed)
    """,
    """
    CREATE INDEX memory_superseded IF NOT EXISTS
    FOR (m:Memory)
    ON (m.superseded)
    """,
    """
    CREATE INDEX memory_type IF NOT EXISTS
    FOR (m:Memory)
    ON (m.memory_type)
    """,
]


def setup_schema() -> dict:
    dim = infer_embedding_dimension()

    for q in CONSTRAINTS:
        run_query(q)

    for q in INDEXES:
        run_query(q)

    run_query(f"""
    CREATE VECTOR INDEX memory_embedding_index IF NOT EXISTS
    FOR (m:Memory) ON (m.embedding)
    OPTIONS {{
      indexConfig: {{
        `vector.dimensions`: {dim},
        `vector.similarity_function`: 'cosine'
      }}
    }}
    """)

    return {
        "status": "ready",
        "schema_ready": True,
        "vector_index_ready": True,
        "embedding_dimension": dim,
        "message": "Schema and vector index are ready.",
    }


def cleanup_legacy_schema() -> dict:
    run_query("""
    MATCH (m:Memory)
    WHERE m.quality_score IS NULL
    SET
      m.quality_score = 0.5,
      m.quality_passed = true,
      m.superseded = false,
      m.novelty = 0.8,
      m.max_similarity = 0.0,
      m.canonical_text = toLower(m.text)
    """)

    run_query("""
    MATCH (a:Memory)-[r:SUPERCEDES]->(b:Memory)
    MERGE (a)-[:SUPERSEDES]->(b)
    DELETE r
    """)

    return {
        "status": "ok",
        "cleanup_complete": True,
        "message": "Legacy memory schema cleanup completed.",
    }