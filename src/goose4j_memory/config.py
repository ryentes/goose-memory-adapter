import os
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

@dataclass(frozen=True)
class Settings:
    neo4j_uri: str | None
    neo4j_user: str | None
    neo4j_password: str | None
    neo4j_database: str
    ollama_base_url: str
    ollama_embed_model: str
    log_level: str

    @classmethod
    def from_env(cls) -> "Settings":
        return cls(
            neo4j_uri=os.getenv("NEO4J_URI"),
            neo4j_user=os.getenv("NEO4J_USER"),
            neo4j_password=os.getenv("NEO4J_PASSWORD"),
            neo4j_database=os.getenv("NEO4J_DATABASE", "neo4j"),
            ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://localhost:11434"),
            ollama_embed_model=os.getenv("OLLAMA_EMBED_MODEL", "nomic-embed-text"),
            log_level=os.getenv("LOG_LEVEL", "INFO").upper(),
        )

    def validate(self) -> None:
        missing: list[str] = []
        if not self.neo4j_uri:
            missing.append("NEO4J_URI")
        if not self.neo4j_user:
            missing.append("NEO4J_USER")
        if not self.neo4j_password:
            missing.append("NEO4J_PASSWORD")

        if missing:
            raise RuntimeError(
                "Missing required environment variables: "
                + ", ".join(missing)
                + ". Set them in the shell or in a .env file."
            )


settings = Settings.from_env()
settings.validate()