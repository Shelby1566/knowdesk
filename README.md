# KnowDesk

KnowDesk 是银行工单代理的执行与评测骨架：面包屑混合检索、失败关闭的动态 Schema Guard，以及可复现的 τ-bench 对照。

`Python >=3.12` · `FastAPI` · `LangGraph` · `SQLite` · `MCP` · `OpenTelemetry` · `FAISS` · `LiteLLM`

包版本 `0.1.0`。本文由 `eval/render_readme.py` 在 `2026-11-20T11:41:45Z` 生成。表里的数字只从当时读到的 `RESULTS_*.md` 单元格抄出。

## 架构

```mermaid
flowchart TB
  subgraph evalPath ["τ 评测"]
    runner[eval/tau_runner.py]
    adapter[adapters/tau2_adapter.py]
    agents[knowdesk_plain / knowdesk_graph]
    guard[GuardEngine]
    hybrid[KnowDeskHybridRetriever.search_sections]
    tauTools[τ orchestrator]
    runner --> adapter --> agents
    agents --> guard
    agents --> hybrid
    agents --> tauTools
  end
  subgraph servicePath ["FastAPI 演示"]
    api[service/app.py]
    graph[agent/graph.py]
    confirm[human_confirm interrupt]
    exec[tool_exec]
    ledger[SQLite idempotency_ledger]
    mcp[mcp_client Client]
    otel[OpenTelemetry SDK]
    api --> graph --> confirm --> exec --> ledger
    exec --> mcp
    api --> otel
  end
```

评测路径里的工具由 τ orchestrator 执行。`knowdesk_graph` 在适配器里另编一张每回合的图，不调用 `build_knowdesk_graph`。
演示路径进入 `build_knowdesk_graph`：`agent`、`guard`、`human_confirm`、`tool_exec`。人工确认只 `interrupt()`，恢复时不重跑已经执行过的工具。
服务在未同时传入 `llm_fn` 与 `tool_funcs` 时，用 LiteLLM、混合检索和进程内 MCP。`service.app:app` 第一次被访问时才 `create_knowdesk_app()`。
在线默认读自 `kb/retrieval_defaults.json`：fusion_mode `rrf`，beta `0.15`，selected_on `dev_colloquial`。
镜像把 tau2-bench 固定在 checkout `033bf08`。

| 路径 | 入口 | 工具谁来执行 | 状态放在哪 |
|---|---|---|---|
| τ 评测 | `eval/tau_runner.py` | τ orchestrator | 适配器进程内的安全状态，图用 `InMemorySaver` |
| 无框架循环 | 调用方自己建 `PlainAgentLoop` | `ReliableToolExecutor` | 调用方决定要不要 SQLite 账本 |
| FastAPI 演示 | `service.app:app` | 进程内 MCP | `SqliteSaver` 加 `idempotency_ledger` |

## 实测结果

下面三张表是首页上的全部实验数字。原始 JSON 和完整报告走相对链接，不铺在这里。

### X1 检索消融（test 67）

超参只在 dev 口语查询上选定。这里只抄 test 67 的口语查询和原始工单说明，列是 Recall@10 与 MRR。

| 切分 | 检索配置 | Recall@10 | MRR |
|---|---|---:|---:|
| test 口语 n=67 | τ 原生 BM25Retriever | 0.0774 | 0.2152 |
| test 口语 n=67 | 静态 RRF（同条件） | 0.2276 | 0.5254 |
| test 口语 n=67 | 线性融合 alpha=0.3（dev 上该族最优，同条件） | 0.2547 | 0.5161 |
| test 口语 n=67 | AdaRRF temperature=2.0（dev 上该族最优，同条件） | 0.2204 | 0.5105 |
| test 口语 n=67 | 在线默认（dev 冻结配置，无覆盖参数） | 0.2327 | 0.5173 |
| test 口语 n=67 | 在线默认 Top-30 + bge-reranker-v2-m3（消融，不进入默认） | 0.2192 | 0.5326 |
| test 原始 n=67 | τ 原生 BM25Retriever | 0.1283 | 0.3134 |
| test 原始 n=67 | 静态 RRF（同条件） | 0.2956 | 0.6747 |
| test 原始 n=67 | 线性融合 alpha=0.3（dev 上该族最优，同条件） | 0.3103 | 0.7071 |
| test 原始 n=67 | AdaRRF temperature=2.0（dev 上该族最优，同条件） | 0.2740 | 0.6872 |
| test 原始 n=67 | 在线默认（dev 冻结配置，无覆盖参数） | 0.3058 | 0.7015 |

原文：[`RESULTS_X1.md`](RESULTS_X1.md)。

### X2 银行域端到端（dev 30）

| 配置 | Pass^1 | 工具名召回 | 精确参数召回 | 在线文档召回 | 平均输入 token | 相对基线 Pass^1 差 95% CI |
|---|---:|---:|---:|---:|---:|---|
| τ 原生基线 (llm_agent + 内置 BM25) | 0.1000 | 0.7303 | 0.5327 | 0.7364 | 1159547 | n/a（对照） |
| KnowDesk-Plain (knowdesk_plain + 自研混合检索 + Fail-Closed Guard) | 0.0667 | 0.7321 | 0.5638 | 0.8535 | 883186 | [-0.1333, 0.0667] INCONCLUSIVE |
| KnowDesk-LangGraph (knowdesk_graph + 自研混合检索 + Fail-Closed Guard) | 0.1000 | 0.8092 | 0.6153 | 0.7794 | 770005 | [-0.1000, 0.1000] INCONCLUSIVE |

原文：[`RESULTS_X2.md`](RESULTS_X2.md)。

### X9 中文零售与 X10 规则拦截

| 配置 | DB Match | DB 任务数 | 平均步数 | Infra Errors |
|---|---:|---:|---:|---:|
| llm_agent | 0.4667 | 30 | 19.0000 | 0 |
| knowdesk_graph | 0.5333 | 30 | 17.8333 | 0 |

原文：[`RESULTS_X9_RETAIL_ZH.md`](RESULTS_X9_RETAIL_ZH.md)。

#### X10 表 A：单元级规则拦截

Guard OFF 的违规率按定义记为 `1.0`，因为这条路径不调用 `GuardEngine`。它不是大模型攻击成功率。

| 口径 | 条数 | 放行数 | 违规率 | 规则拦截率 |
|---|---:|---:|---:|---:|
| 混淆 / Base64 载荷，Guard ON | 24 | 0 | 0.0000 | 1.0000 |
| 未核身写调用，Guard ON | 6 | 0 | 0.0000 | 1.0000 |

规则 ID 计数：`{"R02_PROMPT_INJECTION_PAYLOAD": 24, "R03_REQUIRE_UNLOCK_BEFORE_DISCOVERABLE_CALL": 6}`。

原文：[`RESULTS_SAFETY_AND_JUDGE.md`](RESULTS_SAFETY_AND_JUDGE.md)。

## 工程要点

- 面包屑混合检索：BM25 块与 Qwen 向量做静态 RRF，在线默认带意图加权；重排只留在消融里。
- Fail-Closed 动态 Schema Guard：解锁后按当次 schema 校验参数，违规在重试上限后返回固定拒绝并去掉 tool calls。
- LangGraph 把人工确认放在独立 `interrupt` 节点。确认恢复不会把同一写操作再执行一遍。
- SQLite 幂等账本按会话、工具名和规范化参数做 SHA-256。业务失败不写入，进程重启后成功写仍可重放。
- 工具传输用 MCP Python SDK 的 `Client(..., mode="legacy")`。追踪用 OpenTelemetry SDK；只有设置了 OTLP 端点才把 span 送到 Jaeger。

## 演示台工具

在线模型只看到下面这些函数。隐藏写操作走 `call_discoverable_agent_tool`，不逐个放进 schema。

- `KB_search`
- `unlock_discoverable_agent_tool`
- `call_discoverable_agent_tool`
- `give_discoverable_user_tool`
- `get_user_information_by_id`
- `get_user_information_by_name`
- `get_user_information_by_email`
- `get_credit_card_accounts_by_user`
- `get_credit_card_transactions_by_user`
- `get_referrals_by_user`
- `get_current_time`
- `list_discoverable_agent_tools`
- `log_verification`

## Quick Start

```bash
uv run --no-sync pytest -v -W ignore
uv run --no-sync uvicorn service.app:app --host 0.0.0.0 --port 8000
```

演示台是 [http://localhost:8000](http://localhost:8000)。需要 `DASHSCOPE_API_KEY`。单测不需要。

```bash
uv run --no-sync python eval/run_x2.py
uv run --no-sync python eval/run_x9_retail_zh.py
uv run --no-sync python eval/injection.py --table-a
uv run --no-sync python eval/render_readme.py
```

X2 是 dev 30、max_steps=60、三组同条件。X9 是中文 retail 前 30 单的 DB Match。两条都会调用百炼。X1 的 test 67 报告已经在 `RESULTS_X1.md`，生成器不重跑它。
生成时的收集摘要：72 tests collected in 9.54s。解析结果 collected `72`。这是收集数量，不是通过数量。

## 评测边界与诚实局限（Honest Limitations）

- 冻结切分在 `eval/splits.json`：seed `42`，总数 `97`，dev `30`，test `67`，regression `12`。首页数字只来自链接到的 `RESULTS_*.md`；文件写明未跑，表就空着。
- Guard 的失败关闭是 `max_guard_retries = 2` 之后的固定拒绝。它不是对任意输入的拦截率。X10 表 A 是规则单测，Guard OFF 的违规率按定义是 1.0。
- `service/app.py` 与 `adapters/tau2_adapter.py` 都不构造 `PlainAgentLoop`。在线检索不调用 `search_with_subagent_isolation`。
- 跨度里的脱敏是 `security.dlp.redact_text`。没有写进上表的通过率、费用或 Cohen's κ，就是这次发布没有那次测量。

## 结果文件

- [`RESULTS_X4_X7_ABLATION.md`](RESULTS_X4_X7_ABLATION.md)。 X5 尚未在线运行。这里不填写 Pass^1。
- [`eval/results/x10a_manifest.json`](eval/results/x10a_manifest.json)
- [`eval/results/x1_manifest.json`](eval/results/x1_manifest.json)
- [`eval/results/x2_manifest.json`](eval/results/x2_manifest.json)
- [`eval/results/x7_manifest.json`](eval/results/x7_manifest.json)
- [`eval/results/x9_manifest.json`](eval/results/x9_manifest.json)

重新生成：`uv run --no-sync python eval/render_readme.py`。
