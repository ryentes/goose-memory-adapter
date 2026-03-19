import requests

from .config import settings


def embed(text: str) -> list[float]:
    url = f"{settings.ollama_base_url}/api/embeddings"
    resp = requests.post(
        url,
        json={
            "model": settings.ollama_embed_model,
            "prompt": text,
        },
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()["embedding"]


def infer_embedding_dimension() -> int:
    return len(embed("dimension probe"))