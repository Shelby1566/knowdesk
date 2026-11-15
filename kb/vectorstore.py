import json
from pathlib import Path
from typing import Literal
import faiss
import numpy as np

IndexType = Literal["flat", "hnsw"]

def l2_normalize(vecs: np.ndarray) -> np.ndarray:
    arr = np.asarray(vecs, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-12)
    return arr / norms

def create_faiss_index(dim: int, index_type: IndexType = "flat", hnsw_m: int = 32) -> faiss.Index:
    """
    向量索引工厂：
    - 'flat': IndexFlatIP 精确内积检索（适合万级以内块规模，100% 召回无损，耗时 ~0.1ms）
    - 'hnsw': IndexHNSWFlat 图近似检索（适合十万至千万级规模扩展，$O(\\log N)$ 复杂度）
    """
    if index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 64
        idx.hnsw.efSearch = 32
        return idx
    return faiss.IndexFlatIP(dim)

class FaissVectorStore:
    def __init__(self, dim: int, index_type: IndexType = "flat"):
        self.dim = dim
        self.index_type: IndexType = index_type
        self.index = create_faiss_index(dim, index_type=index_type)
        self.item_ids: list[str] = []

    def add(self, item_ids: list[str], vectors: np.ndarray) -> None:
        normed = l2_normalize(vectors)
        assert normed.shape == (len(item_ids), self.dim)
        self.index.add(normed)
        self.item_ids.extend(item_ids)

    def search(self, query_vec: np.ndarray, k: int = 10) -> list[tuple[str, float]]:
        if len(self.item_ids) == 0:
            return []
        q_norm = l2_normalize(query_vec)
        top_k = min(k, len(self.item_ids))
        scores, indices = self.index.search(q_norm, top_k)
        results: list[tuple[str, float]] = []
        for idx, sc in zip(indices[0], scores[0]):
            if 0 <= idx < len(self.item_ids):
                results.append((self.item_ids[idx], float(sc)))
        return results

    def save(self, dir_path: Path, meta: dict | None = None) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        payload = {"dim": self.dim, "index_type": self.index_type, "item_ids": self.item_ids, "meta": meta or {}}
        (dir_path / "meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, dir_path: Path, expected_meta: dict | None = None) -> "FaissVectorStore | None":
        idx_file = dir_path / "index.faiss"
        meta_file = dir_path / "meta.json"
        if not idx_file.exists() or not meta_file.exists():
            return None
        payload = json.loads(meta_file.read_text(encoding="utf-8"))
        if expected_meta and payload.get("meta") != expected_meta:
            return None
        store = cls(dim=int(payload["dim"]), index_type=payload.get("index_type", "flat"))
        store.index = faiss.read_index(str(idx_file))
        store.item_ids = list(payload["item_ids"])
        return store
