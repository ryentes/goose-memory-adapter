import re
from typing import Any

from .helpers import normalize_text
from .mtypes import MemoryType

TRANSIENT_PATTERNS = [
    r"\bthanks\b",
    r"\bok\b",
    r"\bright now\b",
    r"\bthis session\b",
]

PREFERENCE_HINTS = [
    "prefer",
    "likes",
    "dislikes",
    "avoid",
    "prioritize",
    "always",
    "use",
    "include",
    "remember",
    "value",
    "believe",
    "means",
    "principle",
]

PROJECT_HINTS = [
    "ollama",
    "neo4j",
    "database",
    "setup",
    "project",
    "workflow",
    "task",
    "tasks",
    "todo",
    "requirements",
]

ENTITY_SUBJECT_HINTS = [
    "user",
    "richard",
]

ENTITY_ATTRIBUTE_HINTS = [
    "name",
    "background",
    "experience",
    "degree",
    "role",
    "title",
    "field",
    "specialty",
    "occupation",
    "credential",
]

ENTITY_PREDICATE_PATTERNS = [
    r"\buser\s+is\b",
    r"\buser\s+has\b",
    r"\buser\s+works\b",
    r"\buser\s+earned\b",
    r"\buser\s+holds\b",
    r"\buser's\b",
    r"\brichard\s+is\b",
    r"\brichard\s+has\b",
    r"\brichard\s+works\b",
    r"\brichard\s+earned\b",
    r"\brichard\s+holds\b",
    r"\brichard's\b",
]

ABSTRACT_DURABLE_TERMS = [
    "framework",
    "principle",
    "concept",
    "idea",
    "meaning",
    "wisdom",
    "equanimity",
    "grace",
    "kindness",
    "virtue",
    "philosophy",
    "philosophical",
    "synthesis",
    "reflection",
]


def novelty_from_similarity(max_similarity: float) -> float:
    if max_similarity >= 0.97:
        return 0.0
    if max_similarity >= 0.94:
        return 0.1
    if max_similarity >= 0.90:
        return 0.25
    if max_similarity >= 0.85:
        return 0.45
    if max_similarity >= 0.75:
        return 0.65
    return 0.9


def score_relevance(text: str, memory_type: MemoryType) -> float:
    score = 0.3
    lower = text.lower()

    if memory_type == "entity":
        if any(h in lower for h in ENTITY_SUBJECT_HINTS):
            score += 0.2

        if any(h in lower for h in ENTITY_ATTRIBUTE_HINTS):
            score += 0.2

        if any(re.search(pattern, lower) for pattern in ENTITY_PREDICATE_PATTERNS):
            score += 0.25

        return min(score, 1.0)

    if any(x in lower for x in PREFERENCE_HINTS):
        score += 0.3

    if any(x in lower for x in PROJECT_HINTS):
        score += 0.3

    if any(x in lower for x in ABSTRACT_DURABLE_TERMS):
        score += 0.2

    return min(score, 1.0)


def score_specificity(text: str, memory_type: MemoryType) -> float:
    words = text.split()
    score = 0.2

    if len(words) > 4:
        score += 0.2
    if len(words) > 8:
        score += 0.2

    if re.search(r"\d", text):
        score += 0.15

    if ":" in text or "/" in text:
        score += 0.1

    if memory_type == "entity":
        if any(h in text.lower() for h in ENTITY_SUBJECT_HINTS):
            score += 0.1
        if any(h in text.lower() for h in ENTITY_ATTRIBUTE_HINTS):
            score += 0.1

    return min(score, 1.0)


def score_durability(text: str, memory_type: MemoryType) -> float:
    lower = text.lower()

    if any(re.search(pattern, lower) for pattern in TRANSIENT_PATTERNS):
        return 0.1

    if memory_type == "task":
        return 0.6

    if memory_type == "episodic":
        return 0.55

    return 0.7


def score_actionability(text: str, memory_type: MemoryType) -> float:
    lower = text.lower()

    actionable_terms = (
        PREFERENCE_HINTS
        + PROJECT_HINTS
        + ["should", "must", "when", "upon", "before", "after", "prioritize"]
    )

    if memory_type == "entity":
        return 0.5

    if memory_type == "task":
        if any(x in lower for x in actionable_terms):
            return 0.9
        return 0.5

    if any(x in lower for x in actionable_terms):
        return 0.8

    if any(x in lower for x in ABSTRACT_DURABLE_TERMS):
        return 0.65

    return 0.3


def score_sensitivity(text: str) -> float:
    lower = text.lower()

    risk_terms = [
        "password",
        "api key",
        "secret",
        "token",
        "ssn",
    ]

    if any(x in lower for x in risk_terms):
        return 1.0

    return 0.1


def quality_evaluate(
    text: str,
    novelty: float,
    memory_type: MemoryType,
) -> dict[str, Any]:
    text = normalize_text(text)

    if len(text.split()) < 3 and memory_type != "entity":
        return {
            "pass": False,
            "score": 0.0,
            "relevance": 0.0,
            "specificity": 0.0,
            "durability": 0.0,
            "actionability": 0.0,
            "novelty": round(novelty, 3),
            "sensitivity": 0.0,
            "memory_type": memory_type,
        }

    relevance = score_relevance(text, memory_type)
    specificity = score_specificity(text, memory_type)
    durability = score_durability(text, memory_type)
    actionability = score_actionability(text, memory_type)
    sensitivity = score_sensitivity(text)

    if memory_type == "entity":
        quality = (
            0.30 * relevance
            + 0.20 * specificity
            + 0.25 * durability
            + 0.15 * novelty
            + 0.10 * actionability
        )
        passed = (
            relevance >= 0.35
            and specificity >= 0.25
            and durability >= 0.6
            and novelty >= 0.2
            and quality >= 0.45
            and sensitivity < 0.8
        )

    elif memory_type == "task":
        quality = (
            0.25 * relevance
            + 0.15 * specificity
            + 0.15 * durability
            + 0.15 * novelty
            + 0.30 * actionability
        )
        passed = (
            relevance >= 0.35
            and specificity >= 0.3
            and actionability >= 0.5
            and novelty >= 0.1
            and quality >= 0.5
            and sensitivity < 0.8
        )

    elif memory_type == "episodic":
        quality = (
            0.25 * relevance
            + 0.25 * specificity
            + 0.15 * durability
            + 0.20 * novelty
            + 0.15 * actionability
        )
        passed = (
            relevance >= 0.35
            and specificity >= 0.35
            and novelty >= 0.2
            and quality >= 0.5
            and sensitivity < 0.8
        )

    else:  # semantic
        quality = (
            0.30 * relevance
            + 0.20 * specificity
            + 0.20 * durability
            + 0.20 * novelty
            + 0.10 * actionability
        )
        passed = (
            relevance >= 0.4
            and specificity >= 0.35
            and durability >= 0.5
            and novelty >= 0.2
            and quality >= 0.5
            and sensitivity < 0.8
        )

    return {
        "pass": passed,
        "score": round(quality, 3),
        "relevance": round(relevance, 3),
        "specificity": round(specificity, 3),
        "durability": round(durability, 3),
        "actionability": round(actionability, 3),
        "novelty": round(novelty, 3),
        "sensitivity": round(sensitivity, 3),
        "memory_type": memory_type,
    }