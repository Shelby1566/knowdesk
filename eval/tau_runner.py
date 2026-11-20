"""Run a τ-bench configuration and refuse to reuse a cache from a different signature."""

from __future__ import annotations

import builtins
import hashlib
import json
import os
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
TAU_ROOT = ROOT.parent / "tau2-bench"
SIM_ROOT = TAU_ROOT / "data" / "simulations"
DEFAULT_MODEL = "dashscope/qwen3.8-flash"
DEFAULT_LLM_ARGS: dict[str, Any] = {"temperature": 0, "extra_body": {"enable_thinking": False}}
PROTOCOL_MAX_STEPS = 70
INFRA_REASONS = {"infrastructure_error", "unexpected_error"}


def prefer_direct_dashscope() -> None:
    """Reach dashscope.aliyuncs.com directly. A local HTTP proxy drops its TLS handshake."""
    host = "dashscope.aliyuncs.com"
    for name in ("NO_PROXY", "no_proxy"):
        parts = [part.strip() for part in os.environ.get(name, "").split(",") if part.strip()]
        if host not in parts:
            parts.append(host)
        os.environ[name] = ",".join(parts)


def install_direct_dashscope_client() -> None:
    """Replace any HTTP client created before dashscope.aliyuncs.com was on NO_PROXY."""
    prefer_direct_dashscope()
    import httpx
    import litellm

    limits = httpx.Limits(max_keepalive_connections=5, max_connections=10)
    litellm.client_session = httpx.Client(
        limits=limits,
        trust_env=True,
        follow_redirects=True,
        timeout=120.0,
    )


def load_tau_env() -> None:
    prefer_direct_dashscope()
    env_path = TAU_ROOT / ".env"
    if env_path.exists():
        load_dotenv(env_path, override=False)
    prefer_direct_dashscope()
    builtins.input = lambda *_args, **_kwargs: "y"


def require_dashscope_key() -> str:
    load_tau_env()
    key = os.getenv("DASHSCOPE_API_KEY", "")
    if not key or key.startswith("your_"):
        raise RuntimeError("DASHSCOPE_API_KEY is missing; refusing to run a keyword or cached substitute")
    return key


def install_cached_token_patch() -> None:
    """Keep the provider's cached-token count on the message usage dict."""
    import tau2.utils.llm_utils as llm_utils

    current = llm_utils.get_response_usage
    if getattr(current, "_knowdesk_cached", False):
        return
    original = current

    def patched(response: Any) -> dict[str, Any] | None:
        usage = original(response)
        if usage is None:
            return None
        raw = response.get("usage") if hasattr(response, "get") else None
        details = getattr(raw, "prompt_tokens_details", None) if raw is not None else None
        if details is None and isinstance(raw, dict):
            details = raw.get("prompt_tokens_details")
        cached = details.get("cached_tokens") if isinstance(details, dict) else getattr(details, "cached_tokens", None)
        if cached is not None:
            usage["cached_tokens"] = int(cached)
        return usage

    patched._knowdesk_cached = True  # type: ignore[attr-defined]
    llm_utils.get_response_usage = patched


def switch_nl_assertions_to_dashscope(model: str) -> None:
    """Point τ's NL assertion judge at Bailian. The OpenAI default is not used."""
    import tau2.config as config
    import tau2.evaluator.evaluator_nl_assertions as nl_eval

    arguments = {"temperature": 0}
    config.DEFAULT_LLM_NL_ASSERTIONS = model
    config.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(arguments)
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS = model
    nl_eval.DEFAULT_LLM_NL_ASSERTIONS_ARGS = dict(arguments)


def cache_dirname(
    *,
    agent: str,
    retrieval: str,
    prompt_version: str,
    max_steps: int,
    guard_mode: str,
    compaction_mode: str,
    task_ids: list[str],
    num_trials: int,
    domain: str,
    language: str | None,
    protocol: str | None = None,
) -> str:
    """Directory identity is the full run signature, so a different protocol cannot reuse it."""
    digest = hashlib.sha256("\n".join(task_ids).encode("utf-8")).hexdigest()[:8]
    language_tag = language or "default"
    name = (
        f"{domain}_{language_tag}_{agent}_{retrieval}_p{prompt_version}_ms{max_steps}"
        f"_g{guard_mode}_c{compaction_mode}_n{len(task_ids)}_t{num_trials}_{digest}"
    )
    if protocol:
        return f"{name}_{protocol}"
    return name


def _signature(
    *,
    dirname: str,
    domain: str,
    agent: str,
    retrieval: str,
    prompt_version: str,
    max_steps: int,
    guard_mode: str,
    compaction_mode: str,
    task_ids: list[str],
    num_trials: int,
    seed: int,
    model: str,
    llm_args: dict[str, Any],
    language: str | None,
    protocol: str | None = None,
) -> dict[str, Any]:
    payload = {
        "dirname": dirname,
        "domain": domain,
        "agent": agent,
        "retrieval_mode": retrieval,
        "prompt_version": prompt_version,
        "max_steps": max_steps,
        "guard_mode": guard_mode,
        "compaction_mode": compaction_mode,
        "task_ids": list(task_ids),
        "num_trials": num_trials,
        "seed": seed,
        "model": model,
        "llm_args": llm_args,
        "language": language,
    }
    if protocol:
        payload["protocol"] = protocol
    return payload


def _coverage(results_path: Path, task_ids: list[str], num_trials: int) -> None:
    if not results_path.exists():
        raise FileNotFoundError(results_path)
    data = json.loads(results_path.read_text(encoding="utf-8"))
    simulations = data.get("simulations") or []
    infra = [
        str(item.get("task_id"))
        for item in simulations
        if str(item.get("termination_reason") or "") in INFRA_REASONS
    ]
    if infra:
        raise RuntimeError(
            f"{results_path} contains {len(infra)} infrastructure errors ({infra[:8]}). "
            "Delete this signed directory before retrying. Missing rewards are not filled with zeros."
        )
    counts: Counter[str] = Counter()
    for item in simulations:
        reward = (item.get("reward_info") or {}).get("reward")
        if reward is None:
            raise RuntimeError(f"{results_path} task {item.get('task_id')} has no reward")
        counts[str(item.get("task_id"))] += 1
    missing = [task_id for task_id in task_ids if counts[task_id] < num_trials]
    if missing:
        raise RuntimeError(f"{results_path} is missing scored trials for {missing[:8]}")


def run_tau(
    *,
    exp: str,
    domain: str,
    agent: str,
    retrieval: str,
    prompt_version: str,
    guard_mode: str,
    compaction_mode: str,
    task_ids: list[str],
    max_steps: int,
    num_trials: int = 1,
    seed: int = 42,
    model: str | None = None,
    llm_args: dict[str, Any] | None = None,
    language: str | None = None,
    nl_judge_model: str | None = None,
    allow_held_out: bool = False,
    protocol: str | None = None,
    extra_argv: list[str] | None = None,
) -> dict[str, Any]:
    if not task_ids:
        raise RuntimeError("refusing to run an empty task list")
    if domain == "banking_knowledge" and not allow_held_out:
        from eval.gate import load_splits, require_tunable

        require_tunable(task_ids, load_splits())
    require_dashscope_key()
    install_cached_token_patch()
    if nl_judge_model:
        switch_nl_assertions_to_dashscope(nl_judge_model)

    used_model = model or os.getenv("KNOWDESK_MODEL", DEFAULT_MODEL)
    used_args = dict(DEFAULT_LLM_ARGS if llm_args is None else llm_args)
    dirname = cache_dirname(
        agent=agent,
        retrieval=retrieval,
        prompt_version=prompt_version,
        max_steps=max_steps,
        guard_mode=guard_mode,
        compaction_mode=compaction_mode,
        task_ids=task_ids,
        num_trials=num_trials,
        domain=domain,
        language=language,
        protocol=protocol,
    )
    signature = _signature(
        dirname=dirname,
        domain=domain,
        agent=agent,
        retrieval=retrieval,
        prompt_version=prompt_version,
        max_steps=max_steps,
        guard_mode=guard_mode,
        compaction_mode=compaction_mode,
        task_ids=task_ids,
        num_trials=num_trials,
        seed=seed,
        model=used_model,
        llm_args=used_args,
        language=language,
        protocol=protocol,
    )
    save_dir = SIM_ROOT / dirname
    signature_path = save_dir / "knowdesk_signature.json"
    results_path = save_dir / "results.json"
    canonical = json.dumps(signature, sort_keys=True, ensure_ascii=False)
    if signature_path.exists():
        existing = signature_path.read_text(encoding="utf-8")
        if existing != canonical:
            raise RuntimeError(f"signature mismatch for {dirname}; refusing to read that cache")
    elif results_path.exists():
        raise RuntimeError(f"{results_path} has no KnowDesk signature; refusing to reuse it")
    else:
        save_dir.mkdir(parents=True, exist_ok=True)
        signature_path.write_text(canonical, encoding="utf-8")

    if results_path.exists():
        try:
            _coverage(results_path, task_ids, num_trials)
            return {"results_path": results_path, "signature": signature, "command": None, "reused": True}
        except RuntimeError as exc:
            if "infrastructure errors" in str(exc) or "signature" in str(exc):
                raise
            print(f"resuming incomplete run {dirname}: {exc}", flush=True)

    os.environ["KNOWDESK_RETRIEVAL"] = retrieval
    os.environ["KNOWDESK_PROMPT_VERSION"] = "v2" if prompt_version == "tau" else prompt_version
    os.environ["KNOWDESK_GUARD"] = guard_mode
    os.environ["KNOWDESK_COMPACTION"] = compaction_mode
    from adapters.tau2_adapter import patch_retrieval_mode, register_knowdesk_components

    register_knowdesk_components()
    patch_retrieval_mode("hybrid" if retrieval == "hybrid" else "builtin")

    argv = [
        "tau2", "run",
        "--domain", domain,
        "--agent", agent,
        "--agent-llm", used_model,
        "--user-llm", used_model,
        "--agent-llm-args", json.dumps(used_args),
        "--user-llm-args", json.dumps(used_args),
        "--max-steps", str(max_steps),
        "--num-trials", str(num_trials),
        "--max-concurrency", "6",
        "--seed", str(seed),
        "--save-to", dirname,
        "--task-ids", *task_ids,
        "--auto-resume",
    ]
    if domain == "banking_knowledge":
        argv.extend(["--retrieval-config", "bm25"])
    if language:
        argv.extend(["--language", language])
    if extra_argv:
        argv.extend(extra_argv)
    from tau2.cli import main as tau2_cli_main

    install_direct_dashscope_client()
    print(f"running {dirname}", flush=True)
    old_argv = sys.argv
    try:
        sys.argv = argv
        tau2_cli_main()
    except SystemExit as exc:
        if exc.code not in (0, None):
            raise RuntimeError(f"tau2 exited {exc.code} for {dirname}") from exc
    finally:
        sys.argv = old_argv
    _coverage(results_path, task_ids, num_trials)
    return {
        "results_path": results_path,
        "signature": signature,
        "command": " ".join(argv),
        "reused": False,
    }


def signed_results_path(
    *,
    domain: str,
    agent: str,
    retrieval: str,
    prompt_version: str,
    guard_mode: str,
    compaction_mode: str,
    task_ids: list[str],
    max_steps: int,
    num_trials: int,
    language: str | None,
    protocol: str | None = None,
) -> Path:
    """Locate a completed run. A missing signature is not treated as a usable cache."""
    dirname = cache_dirname(
        agent=agent,
        retrieval=retrieval,
        prompt_version=prompt_version,
        max_steps=max_steps,
        guard_mode=guard_mode,
        compaction_mode=compaction_mode,
        task_ids=task_ids,
        num_trials=num_trials,
        domain=domain,
        language=language,
        protocol=protocol,
    )
    directory = SIM_ROOT / dirname
    results_path = directory / "results.json"
    signature_path = directory / "knowdesk_signature.json"
    if not results_path.exists():
        raise FileNotFoundError(
            f"{results_path} is missing. Run the signed τ configuration before reading it."
        )
    if not signature_path.exists():
        raise FileNotFoundError(f"{results_path} has no KnowDesk signature; refusing to read it")
    return results_path
