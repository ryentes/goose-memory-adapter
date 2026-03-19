import re
from collections import Counter


def normalize_text(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def canonicalize(text: str) -> str:
    return normalize_text(text.lower())


def normalize_concepts(concepts: list[str] | None) -> list[str]:
    if not concepts:
        return []
    return sorted({c.strip().lower() for c in concepts if c.strip()})

GENERIC_CONCEPTS = {
    "setup", "project", "memory", "system", "database", "server", "client",
    "adapter", "config", "configuration", "tool", "tools", "model",
    "models", "query", "queries", "search", "write", "writes", "read",
    "reads", "conversation", "conversations", "session", "sessions",
    "issue", "issues", "problem", "problems", "thing", "things",
    "workflow", "workflows", "environment",
}

STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "for", "to", "of", "in", "on",
    "at", "by", "with", "from", "into", "over", "under", "up", "down",
    "is", "are", "was", "were", "be", "been", "being", "it", "this",
    "that", "these", "those", "as", "if", "then", "than", "so", "we",
    "you", "they", "he", "she", "i", "am", "my", "your", "our", "their",
    "using", "used", "use",
}

TECH_HINTS = {
    "neo4j", "ollama", "goose", "fastmcp", "mcp", "qwen", "qwen3",
    "nomic", "nomic-embed-text", "cuda", "bolt", "synology", "nas",
    "python", "windows", "wsl", "wsl2", "uv", "stdout", "stderr",
}

def extract_concepts(text: str, max_concepts: int = 8) -> list[str]:
    """
    Conservative concept extraction.
    Prefers technical/product identifiers, structured tokens, paths, ports,
    model names, and repeated meaningful nouns. Avoids generic graph pollution.
    """
    text = normalize_text(text)
    lower = text.lower()

    candidates: list[str] = []

    # Structured technical tokens
    structured_patterns = [
        r"\b[a-z]+[a-z0-9_-]*:[a-z0-9][a-z0-9._-]*\b",   # qwen3:14b, bolt:// partly captured later
        r"\b[a-z0-9._-]+\.[a-z0-9._-]+\b",              # goose_neo4j_memory_mcp.py, localhost style parts
        r"\b[a-z]+://[^\s]+\b",                         # URLs / bolt://...
        r"\b\d{2,5}\b",                                 # ports / dimensions if meaningful
        r"\b[a-z]:\\[^\s]+\b",                          # Windows paths
        r"\b[a-z0-9_-]+(?:/[a-z0-9._-]+)+\b",           # unix-ish paths
        r"\b[a-z]+(?:-[a-z0-9]+){1,}\b",                # nomic-embed-text, goose-memory-adapter
        r"\b[a-z]+(?:_[a-z0-9]+){1,}\b",                # snake_case identifiers
    ]

    for pattern in structured_patterns:
        for match in re.findall(pattern, lower, flags=re.IGNORECASE):
            token = match.strip(".,;:()[]{}\"'")
            if len(token) >= 3:
                candidates.append(token)

    # Explicit technical hints if present as words/substrings
    for hint in TECH_HINTS:
        if hint in lower:
            candidates.append(hint)

    # Capitalized-ish / identifier-ish tokens from original text
    raw_tokens = re.findall(r"\b[A-Za-z][A-Za-z0-9._:-]{2,}\b", text)
    for tok in raw_tokens:
        t = tok.lower().strip(".,;:()[]{}\"'")
        if len(t) < 3:
            continue
        if t in STOPWORDS or t in GENERIC_CONCEPTS:
            continue
        # Favor technical-looking or mixed alnum tokens
        if re.search(r"\d", t) or any(ch in t for ch in "._:-/\\"):
            candidates.append(t)

    # Word-level fallback: only keep uncommon-ish tokens
    word_tokens = re.findall(r"\b[a-z][a-z0-9_-]{2,}\b", lower)
    for tok in word_tokens:
        if tok in STOPWORDS or tok in GENERIC_CONCEPTS:
            continue
        if tok in TECH_HINTS:
            candidates.append(tok)

    # Clean + rank
    cleaned = []
    seen = set()

    for token in candidates:
        token = token.strip().lower()
        token = re.sub(r"^[^\w]+|[^\w:./\\-]+$", "", token)
        if not token or len(token) < 3:
            continue
        if token in STOPWORDS or token in GENERIC_CONCEPTS:
            continue
        if token.isdigit() and len(token) < 4:
            # Keep tiny pure numbers out unless you later decide otherwise
            continue
        if token not in seen:
            seen.add(token)
            cleaned.append(token)

    # Simple ranking: technical/structured tokens first
    def concept_rank(token: str) -> tuple[int, int, str]:
        structured = int(bool(re.search(r"[:./\\_-]|\d", token)))
        tech = int(token in TECH_HINTS or any(h in token for h in TECH_HINTS))
        return (-tech, -structured, token)

    cleaned.sort(key=concept_rank)
    return cleaned[:max_concepts]


def select_expansion_concepts(seed_hits: list[dict], max_concepts: int = 12) -> list[str]:
    counts = Counter()
    for hit in seed_hits:
        for c in hit.get("concepts") or []:
            if c:
                counts[c] += 1

    blacklist = GENERIC_CONCEPTS | {
        "user", "assistant", "chat", "text", "local", "primary",
        "preferred", "preference", "memory_type",
    }

    ranked = []
    for concept, count in counts.items():
        if concept in blacklist:
            continue
        # Slightly prefer concepts that appear in multiple seed hits
        score = count
        if re.search(r"[:./\\_-]|\d", concept):
            score += 1
        if concept in TECH_HINTS:
            score += 1
        ranked.append((score, concept))

    ranked.sort(key=lambda x: (-x[0], x[1]))
    return [concept for _, concept in ranked[:max_concepts]]


def rerank_hybrid_results(
    seed_hits: list[dict],
    expanded_hits: list[dict],
    conversation_hits: list[dict],
    top_k: int,
) -> list[dict]:
    combined: dict[str, dict] = {}

    for hit in seed_hits:
        vector_score = float(hit.get("vector_score", 0.0) or 0.0)
        importance = float(hit.get("importance", 0.5) or 0.5)
        quality_score = float(hit.get("quality_score", 0.5) or 0.5)

        combined[hit["id"]] = {
            **hit,
            "source": "vector",
            "shared_concept_count": len(hit.get("concepts") or []),
            "conversation_overlap_count": 0,
            "final_score": (
                0.60 * vector_score
                + 0.20 * importance
                + 0.20 * quality_score
            ),
        }

    seed_ids = set(combined.keys())

    for hit in expanded_hits:
        importance = float(hit.get("importance", 0.5) or 0.5)
        quality_score = float(hit.get("quality_score", 0.5) or 0.5)
        shared = int(hit.get("shared_concept_count", 0) or 0)
        graph_score = min(1.0, 0.28 * shared)

        if hit["id"] in seed_ids:
            combined_hit = combined[hit["id"]]
            combined_hit["shared_concept_count"] = max(
                combined_hit.get("shared_concept_count", 0),
                shared,
            )
            combined_hit["expansion_source"] = "concept"
            combined_hit["final_score"] += 0.12 * graph_score
            continue

        combined[hit["id"]] = {
            **hit,
            "source": "graph_expansion",
            "vector_score": 0.0,
            "conversation_overlap_count": 0,
            "final_score": (
                0.45 * graph_score
                + 0.25 * importance
                + 0.30 * quality_score
            ),
        }

    for hit in conversation_hits:
        importance = float(hit.get("importance", 0.5) or 0.5)
        quality_score = float(hit.get("quality_score", 0.5) or 0.5)
        overlap = int(hit.get("conversation_overlap_count", 0) or 0)
        conversation_score = min(1.0, 0.35 * overlap)

        if hit["id"] in combined:
            combined_hit = combined[hit["id"]]
            combined_hit["conversation_overlap_count"] = max(
                combined_hit.get("conversation_overlap_count", 0),
                overlap,
            )
            combined_hit["conversation_source"] = "conversation"
            combined_hit["final_score"] += 0.15 * conversation_score
            continue

        combined[hit["id"]] = {
            **hit,
            "source": "conversation_expansion",
            "vector_score": 0.0,
            "shared_concept_count": 0,
            "final_score": (
                0.45 * conversation_score
                + 0.25 * importance
                + 0.30 * quality_score
            ),
        }

    ranked = sorted(combined.values(), key=lambda x: x["final_score"], reverse=True)
    return ranked[:top_k]