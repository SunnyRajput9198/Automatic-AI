import uuid
import asyncio
import structlog
from datetime import datetime, timezone
from typing import Optional

from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

from app.core.config import settings
from dotenv import load_dotenv
load_dotenv()

logger = structlog.get_logger()

# text-embedding-3-small output dimension
_EMBED_DIM = 1536
_EMBED_MODEL = "text-embedding-3-small"


class QdrantMemory:

    COLLECTION_NAME = "agent_memory"

    def __init__(self):
        self._oai = None
        if settings.OPENAI_API_KEY:
            self._oai = OpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url="https://aicredits.in/v1",
            )
        else:
            logger.warning(
                "qdrant_memory_no_openai_key",
                reason="OPENAI_API_KEY not set — memory store/search disabled",
            )

        self.client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
        )
        self._ensure_collection()

    # ------------------------------------------------------------------
    # Collection management
    # ------------------------------------------------------------------

    def _ensure_collection(self):
        """
        Create the collection if it doesn't exist.
        If a collection with the wrong vector size exists (e.g. old 384-dim
        MiniLM collection), recreate it with the correct 1536-dim size.
        """
        collections = {c.name for c in self.client.get_collections().collections}

        if self.COLLECTION_NAME in collections:
            # Verify dimension matches — recreate if stale
            info = self.client.get_collection(self.COLLECTION_NAME)
            existing_dim = info.config.params.vectors.size  # type: ignore[union-attr]
            if existing_dim != _EMBED_DIM:
                logger.warning(
                    "qdrant_collection_dim_mismatch",
                    existing=existing_dim,
                    expected=_EMBED_DIM,
                    action="recreating collection",
                )
                self.client.delete_collection(self.COLLECTION_NAME)
                collections.discard(self.COLLECTION_NAME)

        if self.COLLECTION_NAME not in collections:
            self.client.create_collection(
                collection_name=self.COLLECTION_NAME,
                vectors_config=VectorParams(size=_EMBED_DIM, distance=Distance.COSINE),
            )
            logger.info("qdrant_collection_created", name=self.COLLECTION_NAME, dim=_EMBED_DIM)

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------

    def _embed(self, text: str) -> Optional[list]:
        """Return the embedding vector for text, or None if unavailable."""
        if not self._oai:
            return None
        try:
            resp = self._oai.embeddings.create(model=_EMBED_MODEL, input=text)
            return resp.data[0].embedding
        except Exception as e:
            logger.error("qdrant_embed_error", error=str(e))
            return None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def store_memory(self, session_id: str, query: str, result: str) -> None:
        text   = f"Query:\n{query}\n\nResult:\n{result}"
        vector = self._embed(text)
        if not vector:
            logger.warning("qdrant_store_skipped", reason="embedding unavailable")
            return

        try:
            self.client.upsert(
                collection_name=self.COLLECTION_NAME,
                points=[
                    PointStruct(
                        id=str(uuid.uuid4()),
                        vector=vector,
                        payload={
                            "session_id": session_id,
                            "query":      query,
                            "result":     result,
                            "created_at": datetime.now(timezone.utc).isoformat(),
                        },
                    )
                ],
            )
            logger.info("qdrant_memory_stored", session_id=session_id, query=query[:60])
        except Exception as e:
            logger.error("qdrant_store_error", error=str(e))

    def search_memory(self, query: str, limit: int = 5) -> list[dict]:
        vector = self._embed(query)
        if not vector:
            logger.warning("qdrant_search_skipped", reason="embedding unavailable")
            return []

        try:
            results = self.client.query_points(
                collection_name=self.COLLECTION_NAME,
                query=vector,
                limit=limit * 3,
            ).points

            now    = datetime.now(timezone.utc)
            ranked = []

            for r in results:
                if r.payload is None:
                    continue

                payload          = dict(r.payload)
                similarity_score = float(getattr(r, "score", 0.0))
                recency_score    = 0.0

                try:
                    created_at = payload.get("created_at")
                    if created_at:
                        dt = datetime.fromisoformat(created_at)
                        if dt.tzinfo is None:
                            dt = dt.replace(tzinfo=timezone.utc)
                        age_days      = (now - dt).days
                        recency_score = max(0.0, 1.0 - age_days / 30.0)
                except Exception:
                    pass

                payload["_score"] = similarity_score * 0.8 + recency_score * 0.2
                ranked.append(payload)

            ranked.sort(key=lambda x: x["_score"], reverse=True)
            return ranked[:limit]

        except Exception as e:
            logger.error("qdrant_search_error", error=str(e))
            return []
