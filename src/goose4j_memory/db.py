from typing import Any
from neo4j import GraphDatabase

from .config import settings

driver = GraphDatabase.driver(
    settings.neo4j_uri,
    auth=(settings.neo4j_user, settings.neo4j_password),
)


def run_query(query: str, params: dict | None = None) -> list[dict[str, Any]]:
    with driver.session(database=settings.neo4j_database) as session:
        result = session.run(query, params or {})
        return [r.data() for r in result]