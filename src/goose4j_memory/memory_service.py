import re
import uuid
from typing import Any

from .db import run_query
from .embeddings import embed
from .helpers import (
    normalize_text,
    canonicalize,
    normalize_concepts,
    extract_concepts,
    select_expansion_concepts,
    rerank_hybrid_results,
)
from .quality import novelty_from_similarity, quality_evaluate
from .mtypes import MemoryType, VALID_MEMORY_TYPES


def store_memory(
    text: str,
    conversation_id: str | None = None,
    concepts: list[str] | None = None,
    memory_type: MemoryType = "semantic",
) -> dict[str, Any]:
    conversation_id = conversation_id or str(uuid.uuid4())
    text = normalize_text(text)
    canonical = canonicalize(text)
    memory_type = memory_type.strip().lower()

    if memory_type not in VALID_MEMORY_TYPES:
        return {
            "status": "error",
            "stored": False,
            "message": f"Invalid memory_type '{memory_type}'. Must be one of {sorted(VALID_MEMORY_TYPES)}",
        }

    extracted_concepts = normalize_concepts(concepts) if concepts else extract_concepts(text)
    embedding = embed(text)

    similar = run_query(
        """
        CALL db.index.vector.queryNodes(
          'memory_embedding_index',
          5,
          $embedding
        )
        YIELD node, score
        WHERE coalesce(node.superseded, false) = false
        RETURN
          node.id AS id,
          score,
          node.text AS text,
          node.memory_type AS memory_type
        ORDER BY score DESC
        """,
        {"embedding": embedding},
    )

    max_sim = max((float(x["score"]) for x in similar), default=0.0)

    if max_sim >= 0.97:
        return {
            "status": "rejected",
            "stored": False,
            "reason": "duplicate",
            "max_similarity": max_sim,
            "memory_type": memory_type,
            "similar": similar[:3],
        }

    novelty = novelty_from_similarity(max_sim)
    q = quality_evaluate(text, novelty, memory_type)

    if not q["pass"]:
        return {
            "status": "rejected",
            "stored": False,
            "reason": "quality_rejected",
            "quality": q,
            "max_similarity": max_sim,
            "memory_type": memory_type,
            "concepts": extracted_concepts,
        }

    memory_id = str(uuid.uuid4())

    params = {
        "memory_id": memory_id,
        "text": text,
        "canonical": canonical,
        "embedding": embedding,
        "quality": q["score"],
        "novelty": novelty,
        "max_sim": max_sim,
        "concepts": extracted_concepts,
        "conversation_id": conversation_id,
        "memory_type": memory_type,
    }

    run_query(
        """
        MERGE (m:Memory {id:$memory_id})
        ON CREATE SET m.created_at = datetime()
        SET
          m.text = $text,
          m.canonical_text = $canonical,
          m.embedding = $embedding,
          m.updated_at = datetime(),
          m.quality_score = $quality,
          m.quality_passed = true,
          m.novelty = $novelty,
          m.max_similarity = $max_sim,
          m.superseded = false,
          m.memory_type = $memory_type

        WITH m, $concepts AS concepts, $conversation_id AS conversation_id

        FOREACH (_ IN CASE WHEN conversation_id IS NOT NULL THEN [1] ELSE [] END |
          MERGE (conv:Conversation {id: conversation_id})
          ON CREATE SET conv.created_at = datetime()
          MERGE (m)-[:FROM_CONVERSATION]->(conv)
        )

        WITH m, concepts
        UNWIND concepts AS c
        MERGE (x:Concept {name:c})
        MERGE (m)-[:MENTIONS]->(x)
        """,
        params,
    )

    for hit in similar:
        score = float(hit["score"])
        old_id = hit["id"]
        if 0.85 <= score < 0.97 and old_id != memory_id:
            run_query(
                """
                MATCH (new:Memory {id: $new_id})
                MATCH (old:Memory {id: $old_id})
                MERGE (new)-[:SUPERSEDES]->(old)
                SET old.superseded = true,
                    old.superseded_at = datetime(),
                    old.updated_at = datetime()
                """,
                {"new_id": memory_id, "old_id": old_id},
            )

    return {
        "status": "stored",
        "stored": True,
        "memory_id": memory_id,
        "memory_type": memory_type,
        "quality_score": q["score"],
        "novelty": novelty,
        "max_similarity": max_sim,
        "concepts": extracted_concepts,
        "conversation_id": conversation_id,
    }


def retrieve_memory(
    query: str,
    top_k: int = 5,
    expand_concepts: bool = True,
    memory_type: MemoryType | None = None,
) -> list[dict[str, Any]]:
    
    if memory_type is not None:
        memory_type = memory_type.strip().lower()
        if memory_type not in VALID_MEMORY_TYPES:
            return [
                {
                    "status": "error",
                    "message": f"Invalid memory_type '{memory_type}'. Must be one of: {', '.join(sorted(VALID_MEMORY_TYPES))}",
                }
            ]

    embedding = embed(query)

    seed_hits = run_query(
        """
        CALL db.index.vector.queryNodes(
          'memory_embedding_index',
          $k,
          $embedding
        )
        YIELD node, score
        WHERE coalesce(node.quality_passed, true) = true
          AND coalesce(node.superseded, false) = false
          AND ($memory_type IS NULL OR node.memory_type = $memory_type)
        OPTIONAL MATCH (node)-[:MENTIONS]->(c:Concept)
        OPTIONAL MATCH (node)-[:FROM_CONVERSATION]->(conv:Conversation)
        RETURN
          node.id AS id,
          node.text AS text,
          node.memory_type AS memory_type,
          score AS vector_score,
          coalesce(node.importance, 0.5) AS importance,
          coalesce(node.quality_score, 0.5) AS quality_score,
          collect(DISTINCT c.name) AS concepts,
          collect(DISTINCT conv.id) AS conversation_ids
        ORDER BY vector_score DESC
        """,
        {
            "embedding": embedding,
            "k": max(top_k * 3, 10),
            "memory_type": memory_type,
        },
    )

    if not seed_hits:
        return []

    seed_concepts = select_expansion_concepts(seed_hits) if expand_concepts else []

    expanded_hits: list[dict[str, Any]] = []
    if seed_concepts:
        expanded_hits = run_query(
            """
            UNWIND $concepts AS concept_name
            MATCH (c:Concept {name: concept_name})<-[:MENTIONS]-(m:Memory)
            WHERE coalesce(m.quality_passed, true) = true
              AND coalesce(m.superseded, false) = false
              AND ($memory_type IS NULL OR m.memory_type = $memory_type)
            OPTIONAL MATCH (m)-[:MENTIONS]->(c2:Concept)
            RETURN
              m.id AS id,
              m.text AS text,
              m.memory_type AS memory_type,
              coalesce(m.importance, 0.5) AS importance,
              coalesce(m.quality_score, 0.5) AS quality_score,
              collect(DISTINCT c2.name) AS concepts,
              count(DISTINCT concept_name) AS shared_concept_count
            ORDER BY shared_concept_count DESC, quality_score DESC
            LIMIT $limit
            """,
            {
                "concepts": seed_concepts,
                "limit": max(top_k * 5, 20),
                "memory_type": memory_type,
            },
        )

    seed_conversation_ids = sorted(
        {
            cid
            for hit in seed_hits
            for cid in (hit.get("conversation_ids") or [])
            if cid
        }
    )

    conversation_hits: list[dict[str, Any]] = []
    if seed_conversation_ids:
        conversation_hits = run_query(
            """
            UNWIND $conversation_ids AS conversation_id
            MATCH (conv:Conversation {id: conversation_id})<-[:FROM_CONVERSATION]-(m:Memory)
            WHERE coalesce(m.quality_passed, true) = true
              AND coalesce(m.superseded, false) = false
              AND ($memory_type IS NULL OR m.memory_type = $memory_type)
            OPTIONAL MATCH (m)-[:MENTIONS]->(c:Concept)
            RETURN
              m.id AS id,
              m.text AS text,
              m.memory_type AS memory_type,
              coalesce(m.importance, 0.5) AS importance,
              coalesce(m.quality_score, 0.5) AS quality_score,
              collect(DISTINCT c.name) AS concepts,
              count(DISTINCT conversation_id) AS conversation_overlap_count
            ORDER BY conversation_overlap_count DESC, quality_score DESC
            LIMIT $limit
            """,
            {
                "conversation_ids": seed_conversation_ids,
                "limit": max(top_k * 5, 20),
                "memory_type": memory_type,
            },
        )

    return rerank_hybrid_results(seed_hits, expanded_hits, conversation_hits, top_k)


def adapter_status() -> dict[str, Any]:
    try:
        run_query("RETURN 1 AS ok")

        mem = run_query("MATCH (m:Memory) RETURN count(m) AS count")
        con = run_query("MATCH (c:Concept) RETURN count(c) AS count")
        idx = run_query(
            """
            SHOW INDEXES
            YIELD name, type
            WHERE name = 'memory_embedding_index'
            RETURN count(*) AS count
            """
        )
        type_counts = run_query(
            """
            MATCH (m:Memory)
            RETURN coalesce(m.memory_type, "unknown") AS memory_type, count(m) AS count
            ORDER BY memory_type
            """
        )

        memory_count = mem[0]["count"] if mem else 0
        concept_count = con[0]["count"] if con else 0
        vector_index_present = bool(idx and idx[0]["count"] > 0)

        return {
            "status": "healthy",
            "adapter_running": True,
            "neo4j_connected": True,
            "vector_index_present": vector_index_present,
            "memory_count": memory_count,
            "concept_count": concept_count,
            "memory_type_counts": type_counts,
            "message": "Neo4j memory adapter is running and connected.",
        }

    except Exception as e:
        return {
            "status": "error",
            "adapter_running": True,
            "neo4j_connected": False,
            "vector_index_present": False,
            "message": "Neo4j memory adapter could not query the database.",
            "error": str(e),
        }