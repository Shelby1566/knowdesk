"""Online retrieval defaults.

X1 may replace `retrieval_defaults.json` after a dev-only sweep.
`KnowDeskHybridRetriever` reads this file on every search, so the served
configuration is the frozen X1 choice rather than a second hard-coded policy.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path

CONFIG_PATH = Path(__file__).resolve().parent / "retrieval_defaults.json"


@dataclass(frozen=True)
class RetrievalDefaults:
    fusion_mode: str = "adaptive_rrf"
    temperature: float = 1.0
    rrf_k: int = 60
    alpha: float = 0.5
    beta: float = 0.0
    use_intent_boost: bool = False
    dense_model: str = "Qwen/Qwen3-Embedding-0.6B"
    candidate_k: int = 50
    selected_on: str = "builtin_until_x1"

    def to_dict(self) -> dict:
        return asdict(self)


def builtin_defaults() -> RetrievalDefaults:
    return RetrievalDefaults()


def load_retrieval_defaults(path: Path = CONFIG_PATH) -> RetrievalDefaults:
    if not path.exists():
        return builtin_defaults()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise RuntimeError(f"retrieval defaults must be a JSON object: {path}")
    known = {item.name for item in fields(RetrievalDefaults)}
    unknown = sorted(set(raw) - known)
    if unknown:
        raise RuntimeError(f"unknown retrieval default keys: {unknown}")
    base = builtin_defaults().to_dict()
    base.update(raw)
    return RetrievalDefaults(**base)


def save_retrieval_defaults(cfg: RetrievalDefaults, path: Path = CONFIG_PATH) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(cfg.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)
