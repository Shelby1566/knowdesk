"""Render README.md from repository files. Do not hand-copy experiment numbers.

Inputs, read at generation time:

- ``eval/results/*_manifest.json``
- repository-root ``RESULTS_*.md``
- ``uv run --no-sync pytest --collect-only -q``
- ``eval/splits.json``, ``kb/retrieval_defaults.json``, ``pyproject.toml``
- wiring anchors copied from the current source lines

Regenerate with ``uv run --no-sync python eval/render_readme.py``.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tomllib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

README_PATH = ROOT / "README.md"
PYTEST_COLLECT = ("uv", "run", "--no-sync", "pytest", "--collect-only", "-q")

BANNED_PHRASES = (
    "零幻觉",
    "100% 安全",
    "100%安全",
    "生产级银行",
    "生产级",
    "Google Cloud Spanner",
    "Cloud Spanner",
    "Vertex",
    "Model Armor",
    "all 24 tests",
)

ABSENT_SOURCES = (
    "agent/swarm.py",
    "agent/reflection.py",
    "agent/model_armor.py",
    "kb/graph.py",
    "service/prefetch.py",
    "service/grpc_service.py",
    "eval/shadow_eval.py",
    "eval/red_team.py",
    "proto/knowdesk.proto",
    "tools/eligibility.py",
)

WIRED_SYMBOLS = (
    ("adapters/tau2_adapter.py", "def register_knowdesk_components"),
    ("adapters/tau2_adapter.py", "def _fail_closed_message"),
    ("adapters/tau2_adapter.py", "knowdesk_section_excerpts"),
    ("adapters/tau2_adapter.py", "KNOWDESK_COMPACTION"),
    ("adapters/tau2_adapter.py", "self.max_guard_retries = 2"),
    ("agent/graph.py", 'builder.add_node("agent"'),
    ("agent/graph.py", 'builder.add_node("guard"'),
    ("agent/graph.py", 'builder.add_node("human_confirm"'),
    ("agent/graph.py", 'builder.add_node("tool_exec"'),
    ("plain/loop.py", "class PlainAgentLoop"),
    ("tools/tool_wrapper.py", "def compute_idempotency_key"),
    ("agent/checkpointer.py", "refusing to fall back to sqlite"),
    ("service/app.py", "text/event-stream"),
    ("service/app.py", "def create_knowdesk_app"),
    ("service/app.py", "cost_rmb=0.0"),
    ("service/app.py", "build_knowdesk_graph("),
    ("mcp_client/client.py", 'mode="legacy"'),
    ("mcp_server/server.py", "def create_knowdesk_mcp_server"),
    ("obs/otel_setup.py", "OTLPSpanExporter"),
    ("obs/otel_setup.py", "class PIIRedactingSpanProcessor"),
    ("security/dlp.py", "def redact_text"),
    ("kb/hybrid_retriever.py", "def search_sections"),
    ("kb/hybrid_retriever.py", "def search_with_subagent_isolation"),
    ("kb/vectorstore.py", 'IndexType = Literal["flat", "hnsw"]'),
)

# Symbol must not appear in these files. The implementation may still exist elsewhere.
NOT_REFERENCED = (
    ("search_with_subagent_isolation", ("adapters/tau2_adapter.py", "service/app.py", "agent/graph.py", "plain/loop.py")),
    ("SensitiveDataVault", ("adapters/tau2_adapter.py", "service/app.py", "obs/otel_setup.py", "agent/graph.py", "plain/loop.py")),
    ("PlainAgentLoop", ("adapters/tau2_adapter.py", "service/app.py")),
    ("build_knowdesk_graph", ("adapters/tau2_adapter.py", "plain/loop.py")),
)


def _read(root: Path, rel: str) -> str:
    path = root / rel
    if not path.is_file():
        raise RuntimeError(f"required source is missing: {rel}")
    return path.read_text(encoding="utf-8")


def source_line(root: Path, rel: str, needle: str) -> str:
    for line in _read(root, rel).splitlines():
        if needle in line:
            return line.strip()
    raise RuntimeError(f"{rel} has no line containing {needle!r}")


def verify_wiring(root: Path) -> None:
    """Fail the render when the prose below would no longer match the tree."""
    for rel, needle in WIRED_SYMBOLS:
        if needle not in _read(root, rel):
            raise RuntimeError(f"wiring anchor missing: {rel} :: {needle}")
    for rel in ABSENT_SOURCES:
        if (root / rel).exists():
            raise RuntimeError(f"source that the README treats as absent is back: {rel}")
    for symbol, rels in NOT_REFERENCED:
        for rel in rels:
            if symbol in _read(root, rel):
                raise RuntimeError(f"{rel} now references {symbol}; update the path boundary")
    adapter = _read(root, "adapters/tau2_adapter.py")
    if "KnowDeskHybridRetriever.get_instance()" not in adapter:
        raise RuntimeError("tau adapter no longer uses the process-global retriever")
    if "search_sections(" not in adapter:
        raise RuntimeError("tau adapter no longer calls search_sections")
    service = _read(root, "service/app.py")
    if "def __getattr__" not in service or 'name == "app"' not in service:
        raise RuntimeError("service.app:app is no longer a lazy create_knowdesk_app()")
    if "request_acl" not in service:
        raise RuntimeError("service no longer isolates retrieval ACL on a ContextVar")


def load_pyproject(root: Path) -> dict[str, Any]:
    with (root / "pyproject.toml").open("rb") as handle:
        data = tomllib.load(handle)
    project = data.get("project") or {}
    backend = ((data.get("tool") or {}).get("uv") or {}).get("build-backend") or {}
    if "version" not in project or "requires-python" not in project:
        raise RuntimeError("pyproject.toml is missing version or requires-python")
    modules = backend.get("module-name")
    if not isinstance(modules, list) or not modules:
        raise RuntimeError("pyproject.toml is missing tool.uv.build-backend.module-name")
    return {
        "version": str(project["version"]),
        "requires_python": str(project["requires-python"]),
        "script": str((project.get("scripts") or {}).get("knowdesk", "")),
        "modules": [str(item) for item in modules],
    }


def load_split_counts(root: Path) -> dict[str, int]:
    splits = json.loads((root / "eval" / "splits.json").read_text(encoding="utf-8"))
    dev = [str(item) for item in splits["dev_ids"]]
    test = [str(item) for item in splits["test_ids"]]
    regression = [str(item) for item in splits["regression_ids"]]
    if len(dev) != len(set(dev)) or len(test) != len(set(test)):
        raise RuntimeError("eval/splits.json contains duplicate ids")
    if set(dev) & set(test):
        raise RuntimeError("eval/splits.json dev and test overlap")
    if not set(regression) <= set(dev):
        raise RuntimeError("eval/splits.json regression ids are not a subset of dev")
    total = int(splits["total_tasks"])
    if len(dev) + len(test) != total:
        raise RuntimeError("eval/splits.json dev plus test does not equal total_tasks")
    return {
        "seed": int(splits["seed"]),
        "total": total,
        "dev": len(dev),
        "test": len(test),
        "regression": len(regression),
    }


def load_manifests(root: Path) -> list[tuple[str, dict[str, Any]]]:
    directory = root / "eval" / "results"
    found: list[tuple[str, dict[str, Any]]] = []
    if not directory.is_dir():
        return found
    for path in sorted(directory.glob("*_manifest.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError(f"{path.name} is not a JSON object")
        found.append((path.name, payload))
    return found


def load_results(root: Path) -> list[tuple[str, str]]:
    found = []
    for path in sorted(root.glob("RESULTS_*.md")):
        found.append((path.name, path.read_text(encoding="utf-8").rstrip() + "\n"))
    if not found:
        raise RuntimeError("no RESULTS_*.md files at the repository root")
    return found


def retrieval_defaults_text(root: Path) -> str | None:
    path = root / "kb" / "retrieval_defaults.json"
    if not path.is_file():
        return None
    return path.read_text(encoding="utf-8").rstrip() + "\n"


def collect_pytest_summary(root: Path) -> str:
    proc = subprocess.run(
        PYTEST_COLLECT,
        cwd=root,
        text=True,
        capture_output=True,
        check=False,
    )
    text = proc.stdout or ""
    if proc.stderr:
        text += ("\n" if text else "") + proc.stderr
    if proc.returncode != 0:
        raise RuntimeError(f"pytest --collect-only failed ({proc.returncode}):\n{text}")
    parse_collect_summary(text)
    return text.strip() + "\n"


def parse_collect_summary(text: str) -> dict[str, int]:
    match = re.search(
        r"(?P<collected>\d+) tests? collected(?:, (?P<skipped>\d+) skipped)?(?:, (?P<errors>\d+) errors?)?",
        text,
    )
    if match is None:
        raise RuntimeError(f"pytest --collect-only did not report a count:\n{text}")
    collected = int(match.group("collected"))
    nodes = [
        line
        for line in text.splitlines()
        if re.match(r"\S+\.py::", line)
    ]
    if len(nodes) != collected:
        raise RuntimeError(
            f"pytest node id count {len(nodes)} != summary count {collected}"
        )
    return {
        "collected": collected,
        "skipped": int(match.group("skipped") or 0),
        "errors": int(match.group("errors") or 0),
        "nodes": len(nodes),
    }


def _fence(language: str, body: str) -> str:
    cleaned = body.replace("\r\n", "\n").strip("\n")
    return f"```{language}\n{cleaned}\n```"


def _manifest_block(name: str, payload: dict[str, Any]) -> str:
    identity_keys = (
        "exp_id",
        "git_sha",
        "timestamp",
        "model",
        "seed",
        "split_name",
        "max_steps",
        "command",
        "git_dirty",
    )
    missing = [key for key in identity_keys if key not in payload]
    if missing:
        raise RuntimeError(f"{name} is missing {missing}")
    task_ids = payload.get("task_ids")
    if isinstance(task_ids, dict):
        task_count = ",".join(f"{key}:{len(value)}" for key, value in task_ids.items())
    elif isinstance(task_ids, list):
        task_count = str(len(task_ids))
    else:
        task_count = "n/a"
    rows = "\n".join(
        f"| `{key}` | `{payload[key]}` |"
        for key in identity_keys
    )
    rendered = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return "\n".join([
        f"### `{name}`",
        "",
        f"任务 id 计数（由 JSON 现场计算）：`{task_count}`。",
        "",
        "| 字段 | 值 |",
        "|---|---|",
        rows,
        "",
        "完整 JSON：",
        "",
        _fence("json", rendered),
        "",
    ])


def _results_block(name: str, body: str) -> str:
    return "\n".join([
        f"### `{name}`",
        "",
        "以下内容与该文件逐字一致。",
        "",
        body.rstrip() + "\n",
    ])


def assert_no_banned_phrases(text: str) -> None:
    hits = [phrase for phrase in BANNED_PHRASES if phrase in text]
    if hits:
        raise RuntimeError(f"README contains banned wording: {hits}")


def build_readme(
    root: Path,
    *,
    pytest_summary: str,
    manifests: list[tuple[str, dict[str, Any]]] | None = None,
    results: list[tuple[str, str]] | None = None,
) -> str:
    verify_wiring(root)
    project = load_pyproject(root)
    splits = load_split_counts(root)
    manifests = load_manifests(root) if manifests is None else manifests
    results = load_results(root) if results is None else results
    counts = parse_collect_summary(pytest_summary)
    defaults = retrieval_defaults_text(root)
    generated_at = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    dockerfile = _read(root, "Dockerfile")
    checkout = re.search(r"checkout\s+([0-9a-f]{7,40})", dockerfile)
    if checkout is None:
        raise RuntimeError("Dockerfile does not pin a tau2-bench checkout")
    cmd = source_line(root, "Dockerfile", "uvicorn")
    compose = _read(root, "docker-compose.yml").rstrip() + "\n"
    modules = ", ".join(f"`{name}`" for name in project["modules"])

    limitations = f"""## 评测边界与诚实局限（Honest Limitations）

本节是阅读后面任何表格之前的边界。实验数字不在这里复述。

- 冻结切分在 `eval/splits.json`。生成时读到 seed `{splits["seed"]}`，任务总数 `{splits["total"]}`，dev `{splits["dev"]}`，test `{splits["test"]}`，regression `{splits["regression"]}`。dev 与 test 无交集，regression 是 dev 的子集。调参入口碰到 test id 时，`eval/gate.py` 抛出 `FrozenSplitError`。
- 召回、延迟、Pass^1、成本、攻击成功率和 Cohen's κ 只出现在下方原样嵌入的 `RESULTS_*.md` 与 `eval/results/*_manifest.json`。结果文件写明尚未运行，或对应 manifest 不存在，就还没有这次实验的数字。
- `docs/FAILURE_TAXONOMY.md` 不是 `eval/run_x2.py` 的输出。本生成器不读取、不转载该文件。
- Guard 规则来自 `agent/guard_rules.yaml`（R01 到 R05）。τ 适配器里的失败关闭行为是：`{source_line(root, "adapters/tau2_adapter.py", "self.max_guard_retries = 2")}`。到达该上限后返回固定拒绝文本，并去掉 tool calls。这条机制不能外推成对任意输入的拦截率。
- 追踪属性脱敏使用 `security.dlp.redact_text`。`SensitiveDataVault` 没有被 τ 适配器、`agent/graph.py`、`plain/loop.py`、`service/app.py` 或 `obs/otel_setup.py` 引用。
- `create_checkpointer` 的 sqlite 后端会 `SqliteSaver.setup()`。postgres 与 redis 只在对应驱动已安装时构造；驱动缺失时抛出 ImportError，源码写明 `{source_line(root, "agent/checkpointer.py", "refusing to fall back to sqlite")}`。
- OpenTelemetry SDK 始终安装内存导出器和 `PIIRedactingSpanProcessor`。OTLP `BatchSpanProcessor` 只在 `OTEL_EXPORTER_OTLP_ENDPOINT` 非空时挂上，并把缺少的 `/v1/traces` 补到路径末尾。服务记录模型调用时，token 取自供应商 usage；费用参数目前是源码中的 `{source_line(root, "service/app.py", "cost_rmb=0.0")}`。
- 在线检索每次调用 `load_retrieval_defaults()`。重排函数存在于 `kb/rerank.py`，但不会因为某次消融召回更高就自动写进默认配置。`search_with_subagent_isolation` 定义在 `kb/hybrid_retriever.py`，τ 适配器、服务、`agent/graph.py` 和 `plain/loop.py` 都不调用它。
- `KNOWDESK_COMPACTION` 不是 `1` 时，送给模型的消息保持原样。为 `1` 时只改模型副本，τ 保存的轨迹不改。
- 服务新建会话的 SQL 默认是 `{source_line(root, "service/app.py", "verified INTEGER NOT NULL DEFAULT 0")}`。审批先核对 `X-Role` 为 supervisor，再核对 SQLite `approval_ledger` 或待审批 `idem_key`。
- 当前仓库没有这些源文件：{ ", ".join(f"`{rel}`" for rel in ABSENT_SOURCES) }。
- 本生成器不启动百炼调用，也不补写缺失实验。
"""

    if defaults is None:
        defaults_section = (
            "`kb/retrieval_defaults.json` 不存在。`load_retrieval_defaults()` 使用 "
            "`kb/retrieval_config.py` 里 `RetrievalDefaults` 的字段默认值。生成器不把这些默认值抄成另一张指标表。\n"
        )
    else:
        defaults_section = "\n".join([
            "`kb/retrieval_defaults.json` 原文如下。在线默认就是这一份，生成器不另做一张超参表。",
            "",
            _fence("json", defaults),
            "",
        ])

    manifest_body = (
        "目录 `eval/results/` 下没有 `*_manifest.json`。\n"
        if not manifests
        else "\n".join(_manifest_block(name, payload) for name, payload in manifests)
    )
    results_body = "\n".join(_results_block(name, body) for name, body in results)
    skip_note = (
        f"收集摘要里另有 skipped `{counts['skipped']}`、errors `{counts['errors']}`。"
        if counts["skipped"] or counts["errors"]
        else "收集摘要没有 skipped 或 errors。"
    )

    parts = [
        "# KnowDesk",
        "",
        f"包版本 `{project['version']}`，Python `{project['requires_python']}`，均读自 `pyproject.toml`。",
        "本文档由 `eval/render_readme.py` 生成。实验数字只来自生成当刻读到的 `RESULTS_*.md` 与 `eval/results/*_manifest.json`。",
        f"生成时间（UTC）`{generated_at}`。手改本文件里的数字会在下次生成时被覆盖。",
        "",
        "```bash",
        "uv run --no-sync python eval/render_readme.py",
        "```",
        "",
        limitations.rstrip(),
        "",
        "## 三条真实路径",
        "",
        "三条路径可以在同一次部署里分别存在。它们不共用一张执行图，工具也不走同一个进程入口。",
        "",
        "```mermaid",
        "flowchart TB",
        '  subgraph pathTau ["1. τ 评测"]',
        "    tauCli[eval/tau_runner.py]",
        "    adapter[adapters/tau2_adapter.py]",
        "    agents[knowdesk_plain 与 knowdesk_graph]",
        "    tauGuard[GuardEngine 失败关闭]",
        "    retriever[进程级 KnowDeskHybridRetriever]",
        "    excerpts[当次 pipeline state 的章节摘录]",
        "    tauTools[τ orchestrator 执行工具]",
        "    tauCli --> adapter --> agents",
        "    agents --> tauGuard",
        "    agents --> retriever --> excerpts",
        "    agents --> tauTools",
        "  end",
        '  subgraph pathStandalone ["2. 独立执行图与无框架循环"]',
        "    graph[agent/graph.py]",
        "    plain[plain/loop.py]",
        "    sharedGuard[GuardEngine]",
        "    idem[ReliableToolExecutor SHA-256 账本]",
        "    ckpt[create_checkpointer sqlite 或 memory]",
        "    graph --> sharedGuard --> idem",
        "    graph --> ckpt",
        "    plain --> sharedGuard",
        "  end",
        '  subgraph pathService ["3. FastAPI SSE 与 MCP"]',
        "    api[service/app.py]",
        "    sse[StreamingResponse text/event-stream]",
        "    mcpClient[mcp_client Client]",
        "    mcpServer[mcp_server]",
        "    otel[obs/otel_setup.py]",
        "    jaeger[OTLP HTTP 仅在端点已设置时]",
        "    api --> graph",
        "    api --> sse",
        "    api --> mcpClient --> mcpServer",
        "    api --> otel --> jaeger",
        "  end",
        "```",
        "",
        "### 1. τ 评测路径",
        "",
        "`eval/tau_runner.py` 启动 τ CLI，并调用 `register_knowdesk_components()`。",
        "未设置 `KNOWDESK_RETRIEVAL` 时，该函数使用 `hybrid`，再调用 `patch_retrieval_mode`。",
        "hybrid 把 `BM25Retriever.retrieve` 换成进程级 `KnowDeskHybridRetriever.get_instance().search_sections`。",
        "命中的 H2 正文写入当次检索 state 的 `knowdesk_section_excerpts`，`get_document_content` 优先返回这份摘录。",
        "这条调用不传 ACL，因此 `can_access` 在上下文为空时放行文档。服务路径才把请求级 `AccessControlContext` 放进 `request_acl`。",
        "注册的两个 agent 是 `knowdesk_plain` 与 `knowdesk_graph`。",
        "`knowdesk_graph` 在适配器内部另编一张每回合的 `StateGraph`：`agent` 到 `guard`，checkpointer 是 `InMemorySaver`。",
        "它不调用 `build_knowdesk_graph`。工具调用仍由 τ 的 orchestrator 执行，不经过 `mcp_client`。",
        "银行业 SOP 只在工具名含 `KB_search` 或 `unlock_discoverable_agent_tool` 时拼到策略前面。",
        "Guard 打开时，银行业才 `enforce_unlock`；`enforce_verification` 跟随 `KNOWDESK_GUARD`。",
        "τ agent 把 `require_human_confirm_on_write` 设为关闭，这样回放不会停在人工审批。",
        "安全状态按会话键缓存在适配器进程内，上限 `_SEC_CACHE_MAX`。",
        "",
        "### 2. 独立执行图与无框架循环",
        "",
        "`build_knowdesk_graph` 的节点是 `agent`、`guard`、`human_confirm`、`tool_exec`。",
        "未传入 checkpointer 时编译结果使用 `InMemorySaver`。传入 sqlite 路径时使用 `SqliteSaver`，并先 `setup()`。",
        "`ReliableToolExecutor` 在提供 `idempotency_db_path` 时把结果写入 SQLite `idempotency_ledger`。",
        "键是 `compute_idempotency_key`：会话、工具名和规范化 JSON 的 SHA-256。异常不写入账本。",
        "未提供数据库路径时，账本只留在进程内的 `OrderedDict`。",
        "`PlainAgentLoop` 使用同一套 `GuardEngine` 与 `ReliableToolExecutor`，不编译 LangGraph，也不打开 checkpointer。",
        "`service/app.py` 与 `adapters/tau2_adapter.py` 都不构造 `PlainAgentLoop`。",
        "服务进程进入的是 `build_knowdesk_graph`。无框架循环要由调用方自己创建，并自己决定是否带上 SQLite 账本。",
        "",
        "### 3. FastAPI SSE 与 MCP",
        "",
        "`service.app:app` 由模块 `__getattr__` 在第一次访问时调用 `create_knowdesk_app()`。",
        "该工厂连接 SQLite checkpointer、审批账本和 `ReliableToolExecutor`，再编译第 2 节的图。",
        "未同时传入 `llm_fn` 与 `tool_funcs` 时，`build_live_runtime()` 会接上 LiteLLM、混合检索和 `MCPToolBackend`。",
        "单元测试若要避免在线客户端，需要两个参数一起传入。",
        "`POST /sessions/{session_id}/messages` 在生产者线程里跑 `graph.stream`，再由异步 `event_stream` 以 `text/event-stream` 写出。",
        "ContextVar 的设置和复位都在该线程，不跨 yield 持有 token。",
        "进程内 MCP 使用官方 `Client(..., mode=\"legacy\")`，每个调用单独建客户端。",
        "设置了 `server_url` 时改为 `Client(server_url)`。服务端工厂是 `create_knowdesk_mcp_server`。",
        "银行环境加载失败时 `_get_banking_env` 抛出异常，不返回替代对象。",
        "追踪入口是 `obs/otel_setup.py` 的 OpenTelemetry SDK。Compose 把 OTLP 端点指到 Jaeger 时，span 经 OTLP HTTP 送出。",
        "未设置该环境变量时，span 留在内存导出器和环形缓冲里。",
        "",
        "## 在线检索配置",
        "",
        defaults_section.rstrip(),
        "",
        "向量索引类型只接受 `flat`（`IndexFlatIP`）和 `hnsw`。后者的构造参数以 `kb/vectorstore.py` 当前源码为准。",
        "",
        "## 评测记录",
        "",
        "下面两节是生成时读入的原文。manifest 里的 `git_sha` 是写出该文件时的 `HEAD`，`git_dirty` 表示当时工作区是否干净。",
        "后续提交不会回填这些字段。",
        "",
        "### Manifest",
        "",
        manifest_body.rstrip(),
        "",
        "### RESULTS 原文",
        "",
        results_body.rstrip(),
        "",
        "## 测试收集",
        "",
        "数量来自生成时执行的 `uv run --no-sync pytest --collect-only -q`。",
        f"摘要解析结果：collected `{counts['collected']}`，节点 id `{counts['nodes']}`。{skip_note}",
        "这里的 collected 是收集数量，不是通过数量。通过、失败和跳过以实际 `pytest` 运行为准。",
        "",
        _fence("text", pytest_summary),
        "",
        "## 运行",
        "",
        f"控制台脚本入口是 `{project['script']}`，指向 `knowdesk:main`。" if project["script"] else "pyproject 没有声明 `knowdesk` 脚本。",
        "`knowdesk serve` 与镜像启动命令都加载 `service.app:app`，因此会走到上面的懒构造。",
        "",
        "```bash",
        "uv run --no-sync pytest -q -W ignore",
        "uv run --no-sync uvicorn service.app:app --host 0.0.0.0 --port 8000",
        "uv run --no-sync knowdesk serve",
        'uv run --no-sync knowdesk search "credit card annual fee"',
        "```",
        "",
        f"镜像把 tau2-bench 固定在 checkout `{checkout.group(1)}`。Dockerfile 中的启动行是：",
        "",
        _fence("text", cmd),
        "",
        "`docker-compose.yml` 原文：",
        "",
        _fence("yaml", compose),
        "",
        "## 包",
        "",
        f"`tool.uv.build-backend.module-name` 列出的模块：{modules}。",
        "策略文档和 τ 任务不在本仓库，随可编辑依赖 `../tau2-bench` 加载。",
        "",
    ]
    text = "\n".join(parts).rstrip() + "\n"
    assert_no_banned_phrases(text)
    return text


def main() -> None:
    summary = collect_pytest_summary(ROOT)
    rendered = build_readme(ROOT, pytest_summary=summary)
    README_PATH.write_text(rendered, encoding="utf-8")
    print(f"wrote {README_PATH}")


if __name__ == "__main__":
    main()
