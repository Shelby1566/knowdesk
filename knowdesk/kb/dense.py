import hashlib
import os
from pathlib import Path
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from knowdesk.kb.vectorstore import FaissVectorStore, l2_normalize

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
CACHE_ROOT = Path(__file__).resolve().parents[2] / ".cache/kb_dense"

QWEN3_QUERY_PREFIX = (
    "Instruct: Given a bank customer support query, retrieve relevant policy passages that answer the query\nQuery: "
)

def compute_corpus_fingerprint(item_ids: list[str], texts: list[str]) -> str:
    h = hashlib.sha256()
    for cid, t in zip(item_ids, texts):
        h.update(cid.encode("utf-8"))
        h.update(t.encode("utf-8"))
    return h.hexdigest()[:16]

class DenseRetriever:
    def __init__(self, model_name: str, store: FaissVectorStore, model: SentenceTransformer | None = None):
        self.model_name = model_name
        self.store = store
        self.model = model
        self.is_qwen3 = "qwen3-embedding" in model_name.lower()

    @classmethod
    def build_or_load(
        cls,
        model_name: str,
        item_ids: list[str],
        texts: list[str],
        cache_dir: Path | None = None,
        batch_size: int = 64,
    ) -> "DenseRetriever":
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = SentenceTransformer(model_name, device=device)
        fp = compute_corpus_fingerprint(item_ids, texts)
        expected_meta = {"model_name": model_name, "fingerprint": fp, "count": len(item_ids)}

        slug = model_name.replace("/", "__")
        target_dir = cache_dir or (CACHE_ROOT / f"{slug}_{fp}")
        loaded_store = FaissVectorStore.load(target_dir, expected_meta=expected_meta)
        if loaded_store is not None:
            return cls(model_name=model_name, store=loaded_store, model=model)

        vecs = model.encode(
            texts, batch_size=batch_size, show_progress_bar=True, convert_to_numpy=True, normalize_embeddings=True
        )
        store = FaissVectorStore(dim=vecs.shape[1])
        store.add(item_ids, vecs)
        store.save(target_dir, meta=expected_meta)
        return cls(model_name=model_name, store=store, model=model)

    def encode_query(self, query: str) -> np.ndarray:
        q_input = f"{QWEN3_QUERY_PREFIX}{query}" if self.is_qwen3 else query
        vec = self.model.encode([q_input], convert_to_numpy=True, normalize_embeddings=True)
        return l2_normalize(vec)

    def search_docs(self, query: str, chunk_to_doc: dict[str, str], k: int = 10) -> list[tuple[str, float]]:
        q_vec = self.encode_query(query)
        raw = self.store.search(q_vec, k=min(len(self.store.item_ids), k * 8))
        seen: set[str] = set()
        results: list[tuple[str, float]] = []
        for cid, sc in raw:
            did = chunk_to_doc.get(cid, cid)
            if did not in seen:
                seen.add(did)
                results.append((did, sc))
                if len(results) >= k:
                    break
        return results
