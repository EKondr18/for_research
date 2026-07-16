"""Qdrant Cloud storage layer.

Two collections are used:
- chunks_collection: one point per text chunk, with the real embedding vector
  and payload (drive_file_id, file_name, page, text).
- meta_collection: one point per source file (dummy 1-dim vector, only the
  payload matters), used to remember which Drive files/versions have already
  been indexed so /reindex only processes new or changed files. This is kept
  in Qdrant rather than on local disk because Render's free-tier filesystem
  is not guaranteed to persist across restarts/redeploys.
"""
import logging
import uuid

from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

logger = logging.getLogger("qdrant_store")

META_VECTOR_SIZE = 2


def file_point_id(drive_file_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"drive-file:{drive_file_id}"))


class QdrantStore:
    def __init__(self, url: str, api_key: str, chunks_collection: str, meta_collection: str):
        self.client = QdrantClient(url=url, api_key=api_key, timeout=60)
        self.chunks_collection = chunks_collection
        self.meta_collection = meta_collection

    def ensure_collections(self, embedding_dim: int) -> None:
        existing = {c.name for c in self.client.get_collections().collections}

        if self.chunks_collection not in existing:
            logger.info("Создаю коллекцию '%s' в Qdrant (dim=%d)", self.chunks_collection, embedding_dim)
            self.client.create_collection(
                collection_name=self.chunks_collection,
                vectors_config=qmodels.VectorParams(size=embedding_dim, distance=qmodels.Distance.COSINE),
            )

        # Qdrant requires a payload index on a field before it can be used in
        # a filter (e.g. delete_chunks_for_file's filter on drive_file_id) --
        # without this, deleting/filtering by drive_file_id fails with a 400.
        # Creating an index that already exists is a harmless no-op, so this
        # runs unconditionally rather than only when the collection is new.
        self.client.create_payload_index(
            collection_name=self.chunks_collection,
            field_name="drive_file_id",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )

        if self.meta_collection not in existing:
            logger.info("Создаю коллекцию '%s' в Qdrant", self.meta_collection)
            self.client.create_collection(
                collection_name=self.meta_collection,
                vectors_config=qmodels.VectorParams(size=META_VECTOR_SIZE, distance=qmodels.Distance.COSINE),
            )

    def get_indexed_files(self) -> dict[str, str]:
        """Returns {drive_file_id: modified_time} for all files already indexed."""
        result: dict[str, str] = {}
        offset = None
        while True:
            points, offset = self.client.scroll(
                collection_name=self.meta_collection,
                limit=200,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )
            for p in points:
                payload = p.payload or {}
                if "drive_file_id" in payload:
                    result[payload["drive_file_id"]] = payload.get("modified_time", "")
            if offset is None:
                break
        return result

    def upsert_file_meta(self, drive_file_id: str, name: str, modified_time: str) -> None:
        self.client.upsert(
            collection_name=self.meta_collection,
            points=[
                qmodels.PointStruct(
                    id=file_point_id(drive_file_id),
                    vector=[0.0, 0.0],
                    payload={"drive_file_id": drive_file_id, "name": name, "modified_time": modified_time},
                )
            ],
        )

    def delete_chunks_for_file(self, drive_file_id: str) -> None:
        self.client.delete(
            collection_name=self.chunks_collection,
            points_selector=qmodels.FilterSelector(
                filter=qmodels.Filter(
                    must=[qmodels.FieldCondition(key="drive_file_id", match=qmodels.MatchValue(value=drive_file_id))]
                )
            ),
        )

    def upsert_chunks(
        self,
        drive_file_id: str,
        file_name: str,
        chunks: list[tuple[str, int]],
        vectors: list[list[float]],
    ) -> None:
        points = []
        for (text, page), vector in zip(chunks, vectors):
            point_id = str(uuid.uuid4())
            points.append(
                qmodels.PointStruct(
                    id=point_id,
                    vector=vector,
                    payload={
                        "drive_file_id": drive_file_id,
                        "file_name": file_name,
                        "page": page,
                        "text": text,
                    },
                )
            )
        if points:
            self.client.upsert(collection_name=self.chunks_collection, points=points)

    def search(self, query_vector: list[float], top_k: int):
        return self.client.query_points(
            collection_name=self.chunks_collection,
            query=query_vector,
            limit=top_k,
            with_payload=True,
        ).points
