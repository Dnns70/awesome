from __future__ import annotations

import chromadb
from chromadb.config import Settings


class VectorStore:
    def __init__(self, path: str) -> None:
        self._client = chromadb.PersistentClient(
            path=path,
            settings=Settings(anonymized_telemetry=False),
        )
        self._collection = self._client.get_or_create_collection(
            name="memories",
            metadata={"hnsw:space": "cosine"},
        )

    def store_memory(self, memory_id: str, content: str, metadata: dict | None = None) -> None:
        # ChromaDB requires non-empty metadata dicts; use a sentinel if none provided
        meta = metadata if metadata else {"_stored": "true"}
        existing = self._collection.get(ids=[memory_id])
        if existing["ids"]:
            self._collection.update(
                ids=[memory_id],
                documents=[content],
                metadatas=[meta],
            )
        else:
            self._collection.add(
                ids=[memory_id],
                documents=[content],
                metadatas=[meta],
            )

    def search(self, query: str, n_results: int = 5) -> list[dict]:
        count = self._collection.count()
        if count == 0:
            return []
        results = self._collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )
        items = []
        for i, doc_id in enumerate(results["ids"][0]):
            items.append({
                "id": doc_id,
                "content": results["documents"][0][i],
                "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                "distance": results["distances"][0][i] if results["distances"] else 0.0,
            })
        return items

    def count(self) -> int:
        return self._collection.count()
