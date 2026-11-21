# KnowDesk

面向银行工单办理场景的知识驱动受控 Agent 框架，集成面包屑混合检索、Fail-Closed 动态 Schema 护栏与 τ-bench 可复现评测体系。

`Python >=3.12` · `FastAPI` · `LangGraph` · `SQLite` · `MCP` · `OpenTelemetry` · `FAISS` · `LiteLLM`

版本 `0.1.0`。文档由 `eval/render_readme.py` 于 `2026-11-21T15:48:32Z` 自动构建，各项基准指标严格同步自 `RESULTS_*.md` 实验归档。

## 架构设计

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

系统划分两条核心执行路径：
- **τ 评测回路**：工具执行统一委托给 τ-bench orchestrator。适配器中的 `knowdesk_graph` 针对单轮交互维护独立图结构（不调用在线服务态的 `build_knowdesk_graph`），保证评测环境严格隔离。
- **FastAPI 服务回路**：通过 `build_knowdesk_graph` 构建完整的状态机工作流（`agent` → `guard` → `human_confirm` → `tool_exec`）。人工审批节点触发 LangGraph 原生 `interrupt()` 进行零副作用等待，恢复执行时严格幂等，避免重跑已完成工具。
服务默认集成 LiteLLM、混合检索与进程内 MCP；首次访问 `service.app:app` 时触发 `create_knowdesk_app()` 懒加载初始化。
在线混合检索默认配置（`kb/retrieval_defaults.json`）：fusion_mode `rrf`，beta `0.15`，selected_on `dev_colloquial`。
评测环境容器镜像将 tau2-bench 锁定于提交 `033bf08`。

| 路径 | 入口 | 工具谁来执行 | 状态放在哪 |
|---|---|---|---|
| τ 评测 | `eval/tau_runner.py` | τ orchestrator | 适配器进程内的安全状态，图用 `InMemorySaver` |
| 无框架循环 | 调用方自己建 `PlainAgentLoop` | `ReliableToolExecutor` | 调用方决定要不要 SQLite 账本 |
| FastAPI 演示 | `service.app:app` | 进程内 MCP | `SqliteSaver` 加 `idempotency_ledger` |

## 实测结果

### 核心成果一览 (Highlights)

| 核心维度 | τ 原生基线 | KnowDesk 表现 | 相对收益与工程价值 |
|---|---|---|---|
| 📉 **推理成本 (Token)** | 192.9 万 / dev30 | **99.6 万 / dev30** | **输入 Token 骤降 48.4%**（章节级切块去噪） |
| 🎯 **口语检索 (R@10)** | 7.74% (MRR 0.215) | **23.27% (MRR 0.517)** | **Recall 提升 3.0x**，MRR 提升 140% |
| 🛡️ **护栏安检 (Fail-Closed)** | 0% (无护栏) | **100% 规则拦截率** | 30/30 对抗用例 0 放行，30 单端到端 0 误拦 |
| 🏆 **跨域终态 (DB Match)** | 46.67% (retail_zh) | **53.33% (retail_zh)** | 任务成功率绝对值 **+6.66%**，平均步数减少 1.17 步 |

### X1 检索消融（test 67）

基于 `dev 30` 口语查询集调优超参，并在冻结的 `test 67` 独立测试集（含口语化查询与原始工单说明两组切分）上盲测，核心指标为 Recall@10 与 MRR：

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

> **工程结论**：针对全库跨文档重复 H2 导致的上下文主语丢失问题，面包屑切块（文档标题 > H2）与 RRF 融合带来 3 倍口语检索召回提升；实测本地 Reranker 增加约 5s 延迟且未带来显著增益，经数据驱动决策默认不启用重排。

### X2 银行域端到端（dev 30）

| 配置 | Pass^1 | 工具名召回 | 精确参数召回 | 在线文档召回 | 平均输入 token | 相对基线 Pass^1 差 95% CI |
|---|---:|---:|---:|---:|---:|---|
| τ 原生基线 (llm_agent + 内置 BM25) | 0.1333 | 0.8539 | 0.6408 | 0.7657 | 1928673 | n/a（对照） |
| KnowDesk-Plain (knowdesk_plain + 自研混合检索 + Fail-Closed Guard) | 0.1667 | 0.8832 | 0.6802 | 0.8928 | 1112684 | [-0.1333, 0.2000] INCONCLUSIVE |
| KnowDesk-LangGraph (knowdesk_graph + 自研混合检索 + Fail-Closed Guard) | 0.1667 | 0.8957 | 0.6396 | 0.9017 | 995611 | [-0.1000, 0.1667] INCONCLUSIVE |

原文：[`RESULTS_X2.md`](RESULTS_X2.md)。

> **工程结论**：章节级上下文精准注入使端到端多轮 Token 消耗降低 48.4%，文档召回率提升至 90.17%；瓶颈分析表明主要未完成工单受制于轻量非思考模型的数值计算与复合布尔推理极限。

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

> **工程结论**：四层混淆归一化结合 Fail-Closed 动态 Schema 校验，在 30 任务端到端实测 0 误拦的同时，实现了确定性的越权与注入阻断。

## 核心工程要点

- **面包屑混合检索**：结合文档标题与 H2 面包屑切块，使用 BM25 倒排索引与 Qwen 向量进行静态 RRF 融合，在线默认启用产品线意图加权；本地重排经消融评估后保持独立可选。
- **Fail-Closed 动态 Schema 护栏**：工具解锁后按当次动态 schema 校验入参，当重试达到上限仍未通过时，严格返回固定拒绝并剥离 tool calls，阻断越权与非受控执行。
- **人机协同审批与状态隔离**：LangGraph 将人工审批独立为 `interrupt` 节点，会话恢复唤醒时保持幂等，避免重跑已完成的工具操作。
- **持久化幂等账本**：SQLite 账本基于会话、工具名及规范化参数计算 SHA-256 唯一指纹；仅记录成功写操作，服务重启后依然支持重放校验。
- **标准协议与链路可观测**：工具通信遵循 MCP 协议标准（基于 MCP Python SDK `Client(..., mode="legacy")`）；可观测性基于 OpenTelemetry SDK，内置敏感 PII 递归脱敏，按需导出至 OTLP/Jaeger。

## 演示台工具

在线模型暴露以下函数集合。隐藏写操作统一通过 `call_discoverable_agent_tool` 动态调用，不逐个硬编码进入全局 schema：

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

## 快速开始

```bash
# 运行单元测试
uv run --no-sync pytest -v -W ignore

# 启动在线演示服务
uv run --no-sync uvicorn service.app:app --host 0.0.0.0 --port 8000
```

本地服务访问地址：[http://localhost:8000](http://localhost:8000)。在线调用需配置环境变量 `DASHSCOPE_API_KEY`（离线单元测试无需任何 API Key）。

### 实验复现与文档同步

```bash
uv run --no-sync python eval/run_x2.py
uv run --no-sync python eval/run_x9_retail_zh.py
uv run --no-sync python eval/injection.py --table-a
uv run --no-sync python eval/render_readme.py
```

> **评测说明**：X2 对应 dev 30 任务集在同等 max_steps 预算下的对照评测；X9 为中文 retail 域前 30 单的数据库终态匹配测试（两者均调用百炼模型接口）。X1 冻结测试集数据直接归档于 `RESULTS_X1.md`。
> **测试套件状态**：70 tests collected in 9.66s（汇总 collected `70` 项测试用例）。

## 结果文件

- [`RESULTS_X4_X7_ABLATION.md`](RESULTS_X4_X7_ABLATION.md)。 X5 尚未在线运行。这里不填写 Pass^1。
- [`eval/results/x10a_manifest.json`](eval/results/x10a_manifest.json)
- [`eval/results/x1_manifest.json`](eval/results/x1_manifest.json)
- [`eval/results/x2_manifest.json`](eval/results/x2_manifest.json)
- [`eval/results/x7_manifest.json`](eval/results/x7_manifest.json)
- [`eval/results/x9_manifest.json`](eval/results/x9_manifest.json)
