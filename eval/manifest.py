"""Write the trace file that every experiment must leave next to its metrics."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = Path(__file__).resolve().parent / "results"

REQUIRED_FIELDS = (
    "git_sha",
    "timestamp",
    "model",
    "llm_args",
    "seed",
    "split_name",
    "task_ids",
    "max_steps",
    "command",
    "metrics",
)


def git_sha(root: Path = ROOT) -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def write_manifest(
    exp_id: str,
    *,
    model: str,
    llm_args: dict[str, Any],
    seed: int | None,
    split_name: str,
    task_ids: Any,
    max_steps: int | None,
    command: str,
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
    directory: Path | None = None,
) -> Path:
    if not exp_id or "/" in exp_id:
        raise RuntimeError(f"invalid exp_id: {exp_id!r}")
    payload: dict[str, Any] = {
        "exp_id": exp_id,
        "git_sha": git_sha(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model": model,
        "llm_args": llm_args,
        "seed": seed,
        "split_name": split_name,
        "task_ids": task_ids,
        "max_steps": max_steps,
        "command": command,
        "metrics": metrics,
        "git_dirty": bool(
            subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip()
        ),
    }
    if extra:
        overlap = set(extra) & set(payload)
        if overlap:
            raise RuntimeError(f"manifest extra collides with required fields: {sorted(overlap)}")
        payload.update(extra)
    missing = [key for key in REQUIRED_FIELDS if key not in payload]
    if missing:
        raise RuntimeError(f"manifest missing fields: {missing}")
    out_dir = RESULTS_DIR if directory is None else Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{exp_id}_manifest.json"
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path
