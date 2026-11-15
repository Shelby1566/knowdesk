# X9 中文第二域 (`retail --language zh`) 迁移评测结果 (n=10, 模型: `dashscope/qwen3.8-flash`)

| 配置 | DB 终态通过率 (Pass^1) | 平均交互步数 |
|---|---:|---:|
| 1. τ原生基线 (`llm_agent`) | 0.7000 | 34.3 |
| 2. KnowDesk-LangGraph (`knowdesk_graph` + Guard) | 0.5000 | 28.1 |
