import json
from pathlib import Path
from typing import Any, Literal
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
    - 'flat': IndexFlatIP 精确内积检索
    - 'hnsw': IndexHNSWFlat 近似检索
    """
    if index_type == "flat":
        return faiss.IndexFlatIP(dim)
    if index_type == "hnsw":
        idx = faiss.IndexHNSWFlat(dim, hnsw_m, faiss.METRIC_INNER_PRODUCT)
        idx.hnsw.efConstruction = 64
        idx.hnsw.efSearch = 32
        return idx
    raise ValueError(
        f"Unsupported index_type '{index_type}'. FaissVectorStore only supports 'flat' and 'hnsw'."
    )

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

    def search(
        self, query_vec: np.ndarray, k: int = 10, allowed_ids: set[str] | None = None
    ) -> list[tuple[str, float]]:
        if len(self.item_ids) == 0:
            return []
        q_norm = l2_normalize(query_vec)
        if allowed_ids is not None:
            if not allowed_ids:
                return []
            valid_indices = [i for i, cid in enumerate(self.item_ids) if cid in allowed_ids]
            if not valid_indices:
                return []
            top_k = min(k, len(valid_indices))
            # Pre-filtering: compute exact inner product only over ACL-authorized vectors
            # so high-clearance documents cannot crowd out low-clearance results from candidate_k.
            sub_vecs = np.vstack([self.index.reconstruct(i) for i in valid_indices])
            sims = (sub_vecs @ q_norm[0]).astype(np.float32)
            order = np.argsort(-sims)[:top_k]
            return [(self.item_ids[valid_indices[int(pos)]], float(sims[int(pos)])) for pos in order]

        top_k = min(k, len(self.item_ids))
        scores, indices = self.index.search(q_norm, top_k)
        results: list[tuple[str, float]] = []
        for idx, sc in zip(indices[0], scores[0]):
            if idx >= 0 and idx < len(self.item_ids):
                results.append((self.item_ids[idx], float(sc)))
        return results

    def save(self, dir_path: Path, meta: dict[str, Any] | None = None) -> None:
        dir_path.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(dir_path / "index.faiss"))
        payload = {"dim": self.dim, "item_ids": self.item_ids, "index_type": self.index_type}
        if meta:
            payload.update(meta)
        (dir_path / "meta.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, dir_path: Path, expected_meta: dict[str, Any] | None = None) -> "FaissVectorStore | None":
        meta_file = dir_path / "meta.json"
        index_file = dir_path / "index.faiss"
        if not meta_file.exists() or not index_file.exists():
            return None
        meta = json.loads(meta_file.read_text(encoding="utf-8"))
        if expected_meta:
            for k, v in expected_meta.items():
                if meta.get(k) != v:
                    return None
        store = cls(dim=meta["dim"], index_type=meta.get("index_type", "flat"))
        store.index = faiss.read_index(str(index_file))
        store.item_ids = meta["item_ids"]
        return store
