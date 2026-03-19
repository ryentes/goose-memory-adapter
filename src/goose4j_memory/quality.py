from .helpers import normalize_text

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

NOVELTY_THRESHOLDS = [
    (0.97, 0.0),
    (0.94, 0.1),
    (0.90, 0.25),
    (0.85, 0.45),
    (0.75, 0.65),
]

def novelty_from_similarity(max_similarity: float) -> float:
    for threshold, score in NOVELTY_THRESHOLDS:
        if max_similarity >= threshold:
            return score
    return 0.9

def score_relevance(text: str) -> float:
    score = 0.3
    lower = text.lower()

    if any(x in lower for x in PREFERENCE_HINTS):
        score += 0.3

    if any(x in lower for x in PROJECT_HINTS):
        score += 0.3

    abstract_durable_terms = [
        "framework", "principle", "concept", "idea", "meaning",
        "wisdom", "equanimity", "grace", "kindness", "virtue",
        "philosophy", "philosophical", "synthesis", "reflection",
    ]
    if any(x in lower for x in abstract_durable_terms):
        score += 0.2

    return min(score, 1.0)


def score_specificity(text: str) -> float:

    words = text.split()
    score = 0.2

    if len(words) > 8:
        score += 0.3

    if re.search(r"\d", text):
        score += 0.2

    if ":" in text or "/" in text:
        score += 0.1

    return min(score, 1.0)


def score_durability(text: str) -> float:

    lower = text.lower()

    if any(re.search(p, lower) for p in TRANSIENT_PATTERNS):
        return 0.1

    return 0.7


def score_actionability(text: str) -> float:
    lower = text.lower()

    actionable_terms = (
        PREFERENCE_HINTS
        + PROJECT_HINTS
        + ["should", "must", "when", "upon", "before", "after", "prioritize"]
    )

    abstract_guidance_terms = [
        "framework", "principle", "concept", "idea", "wisdom",
        "equanimity", "grace", "kindness", "virtue", "meaning",
        "reflection", "synthesis",
    ]

    if any(x in lower for x in actionable_terms):
        return 0.8

    if any(x in lower for x in abstract_guidance_terms):
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


def quality_evaluate(text: str, novelty: float) -> dict:
    text = normalize_text(text)

    if len(text.split()) < 4:
        return {
            "pass": False,
            "score": 0.0,
            "relevance": 0.0,
            "specificity": 0.0,
            "durability": 0.0,
            "actionability": 0.0,
            "novelty": round(novelty, 3),
            "sensitivity": 0.0,
        }

    relevance = score_relevance(text)
    specificity = score_specificity(text)
    durability = score_durability(text)
    actionability = score_actionability(text)
    sensitivity = score_sensitivity(text)

    quality = (
        0.30 * relevance
        + 0.20 * specificity
        + 0.20 * durability
        + 0.20 * novelty
        + 0.10 * actionability
    )

    passed = (
        relevance >= 0.4
        and specificity >= 0.45
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
    }