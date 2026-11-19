"""Dev-only selection for the X1 retrieval sweep.

The caller must pass rows measured on the dev split. This module does not
load queries and cannot see the test split.
"""

from __future__ import annotations

from typing import Any

ADA_TEMPERATURES = (0.5, 1.0, 2.0)
LINEAR_ALPHAS = (0.3, 0.5, 0.7)
INTENT_BETAS = (0.0, 0.15, 0.30)


def _score(row: dict[str, Any]) -> tuple[float, float, float]:
    return (float(row["Recall@10"]), float(row["MRR"]), -float(row["P50_ms"]))


def _best(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        raise RuntimeError("cannot select from an empty retrieval grid")
    return max(rows, key=_score)


def _require_params(rows: list[dict[str, Any]], field: str, expected: tuple[float, ...]) -> None:
    found = {round(float(row[field]), 4) for row in rows}
    missing = [value for value in expected if round(value, 4) not in found]
    if missing:
        raise RuntimeError(f"dev grid is missing {field}={missing}")


def select_fusion(rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Pick static RRF, AdaRRF temperature, or linear alpha on dev Recall@10."""
    static_rows = [row for row in rows if row.get("family") == "rrf"]
    ada_rows = [row for row in rows if row.get("family") == "adarrf"]
    linear_rows = [row for row in rows if row.get("family") == "linear"]
    if len(static_rows) != 1:
        raise RuntimeError("dev grid must contain exactly one static RRF row")
    _require_params(ada_rows, "temperature", ADA_TEMPERATURES)
    _require_params(linear_rows, "alpha", LINEAR_ALPHAS)
    best_ada = _best(ada_rows)
    best_linear = _best(linear_rows)
    winner = _best([static_rows[0], best_ada, best_linear])
    return {
        "fusion_row": winner,
        "best_adarrf": best_ada,
        "best_linear": best_linear,
        "static_rrf": static_rows[0],
    }


def select_beta(fusion_winner: dict[str, Any], beta_rows: list[dict[str, Any]]) -> dict[str, Any]:
    """Beta 0 is the fusion winner. Positive betas must already be dev measurements."""
    _require_params(beta_rows, "beta", tuple(beta for beta in INTENT_BETAS if beta > 0))
    chosen = _best([fusion_winner, *beta_rows])
    beta = float(chosen.get("beta") or 0.0)
    family = str(fusion_winner["family"])
    if family == "adarrf":
        fusion_mode = "adaptive_rrf"
        temperature = float(fusion_winner["temperature"])
        alpha = 0.5
    elif family == "rrf":
        fusion_mode = "rrf"
        temperature = 1.0
        alpha = 0.5
    elif family == "linear":
        fusion_mode = "linear"
        temperature = 1.0
        alpha = float(fusion_winner["alpha"])
    else:
        raise RuntimeError(f"refusing to freeze fusion family {family}")
    return {
        "fusion_mode": fusion_mode,
        "temperature": temperature,
        "alpha": alpha,
        "beta": beta,
        "use_intent_boost": beta > 0.0,
        "family": family,
        "selected_row": chosen,
        "dev_recall_at_10": float(chosen["Recall@10"]),
        "dev_mrr": float(chosen["MRR"]),
    }
