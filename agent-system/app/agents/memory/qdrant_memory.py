import uuid
from datetime import datetime
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
)

from sentence_transformers import SentenceTransformer
from sympy import limit


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
                        "created_at": datetime.utcnow().isoformat(),
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
            limit=limit * 3,
        ).points

        ranked = []

        now = datetime.utcnow()

        for r in results:

            if r.payload is None:
                continue

            payload = dict(r.payload)

            similarity_score = float(
                getattr(r, "score", 0.0)
            )

            recency_score = 0.0

            try:
                created_at = payload.get(
                    "created_at"
                )

                if created_at:
                    age_days = (
                        now -
                        datetime.fromisoformat(
                            created_at
                        )
                    ).days

                    recency_score = max(
                        0,
                        1 - (age_days / 30)
                    )

            except Exception:
                pass

            final_score = (
                similarity_score * 0.8
                +
                recency_score * 0.2
            )

            payload["_score"] = final_score

            ranked.append(payload)

        ranked.sort(
            key=lambda x: x["_score"],
            reverse=True
        )

        return ranked[:limit]
    
    def summarize_old_memories(self):
        """
        Future feature.
        Summarize memories when count > 50.
        """
        pass