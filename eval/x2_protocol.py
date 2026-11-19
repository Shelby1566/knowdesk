"""Shared factor cells for the dev-30 agent experiments.

Cache directories are named by eval.tau_runner.cache_dirname. The experiment
nickname is not part of that name, so the compaction-off rerun reads the
graph+hybrid cell from X2 instead of paying for it twice.
"""

from __future__ import annotations

PROTOCOL_MAX_STEPS = 70
PROTOCOL_TRIALS = 1
SEED = 42

X2_CELLS: tuple[dict[str, str], ...] = (
    {
        "key": "llm_agent_builtin",
        "label": "llm_agent + builtin（τ 原生基线）",
        "agent": "llm_agent",
        "retrieval": "builtin",
        "prompt_version": "tau",
        "guard_mode": "0",
        "compaction_mode": "0",
    },
    {
        "key": "knowdesk_plain_builtin",
        "label": "knowdesk_plain + builtin",
        "agent": "knowdesk_plain",
        "retrieval": "builtin",
        "prompt_version": "v2",
        "guard_mode": "1",
        "compaction_mode": "0",
    },
    {
        "key": "knowdesk_graph_builtin",
        "label": "knowdesk_graph + builtin",
        "agent": "knowdesk_graph",
        "retrieval": "builtin",
        "prompt_version": "v2",
        "guard_mode": "1",
        "compaction_mode": "0",
    },
    {
        "key": "knowdesk_plain_hybrid",
        "label": "knowdesk_plain + hybrid",
        "agent": "knowdesk_plain",
        "retrieval": "hybrid",
        "prompt_version": "v2",
        "guard_mode": "1",
        "compaction_mode": "0",
    },
    {
        "key": "knowdesk_graph_hybrid",
        "label": "knowdesk_graph + hybrid（v2 + 动态 Schema Guard）",
        "agent": "knowdesk_graph",
        "retrieval": "hybrid",
        "prompt_version": "v2",
        "guard_mode": "1",
        "compaction_mode": "0",
    },
)

BASELINE_KEY = "llm_agent_builtin"
GRAPH_HYBRID_KEY = "knowdesk_graph_hybrid"


def cell_by_key(key: str) -> dict[str, str]:
    for cell in X2_CELLS:
        if cell["key"] == key:
            return dict(cell)
    raise KeyError(key)
