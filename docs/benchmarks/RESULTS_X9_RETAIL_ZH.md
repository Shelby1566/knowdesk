# X9 中文 retail 前 30 单（DB Match）

任务号是中文 retail 语言包的前 30 个（绑定里共 50 条，id 形如 `6_zh_identity`）。
max_steps=50，语言 `zh`，检索 `builtin`。
本表只报告确定性 DB Match。进程内从奖励依据去掉了 `NL_ASSERTION`，communicate 使用 substring，不调用 OpenAI NL 裁判。
τ 在 max_steps 提前结束时不写 db_check。这些轨迹用 EnvironmentEvaluator 重放已保存的工具调用，按数据库哈希补上 DB Match，不改官方 reward，也不再调用模型。
平均步数是轨迹里 assistant 消息条数的均值。重放失败的任务不补 0。

| 配置 | DB Match | DB 任务数 | 平均步数 | Infra Errors |
|---|---:|---:|---:|---:|
| llm_agent | 0.4667 | 30 | 19.0000 | 0 |
| knowdesk_graph | 0.5333 | 30 | 17.8333 | 0 |

## 对照

knowdesk_graph 的 DB Match `0.5333` 高于 llm_agent `0.4667`。
平均步数 knowdesk_graph `17.83`，llm_agent `19.00`。
