import uuid

from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from sentence_transformers import SentenceTransformer


class QdrantMemory:

    COLLECTION_NAME = "agent_memory"

    def __init__(self):
        self.client = QdrantClient(
            host="qdrant",  # Use the service name defined in docker-compose.yml
            port=6333,
        )

        self.embedding_model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self._ensure_collection()

    def _ensure_collection(self):
        collections = self.client.get_collections()

        existing = [
            c.name
            for c in collections.collections
        ]

        if self.COLLECTION_NAME not in existing:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(
                    size=384,
                    distance=Distance.COSINE,
                ),
            )

    def store_memory(
        self,
        session_id: str,
        query: str,
        result: str,
    ):
        text = f"""
Query:
{query}

Result:
{result}
"""

        vector = self.embedding_model.encode(
            text
        ).tolist()

        self.client.upsert(
            collection_name=self.COLLECTION_NAME,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vector,
                    payload={
                        "session_id": session_id,
                        "query": query,
                        "result": result,
                    },
                )
            ],
        )

    def search_memory(
        self,
        query: str,
        limit: int = 5
    ) -> list[dict]:
        vector = self.embedding_model.encode(
            query
        ).tolist()

        results = self.client.query_points(
            collection_name=self.COLLECTION_NAME,
            query=vector,
            limit=limit,
        ).points

        return [
    dict(r.payload)
    for r in results
    if r.payload is not None
]