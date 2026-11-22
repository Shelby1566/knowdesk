# X2 dev 30（max_steps=90，temperature=0，思考关闭，模型 `dashscope/qwen3.8-flash`）

三组使用同一批 dev 30、同一个 max_steps。Pass^1 是任务奖励均值。
步数上限是 90。上一轮 60 步时，knowdesk_graph 有 18/30 条在上限结束。检索、Guard 和模型没有改。
配对 Bootstrap 是候选减 τ 基线，10000 次、种子 42、95% 区间。基线行没有对照差。
按 Pass^1、精确参数召回、工具名召回选出的配置是 `knowdesk_plain_hybrid`。

| 配置 | Pass^1 | 工具名召回 | 精确参数召回 | 在线文档召回 | 平均输入 token | 相对基线 Pass^1 差 95% CI |
|---|---:|---:|---:|---:|---:|---|
| τ 原生基线 (llm_agent + 内置 BM25) | 0.1333 | 0.8539 | 0.6408 | 0.7657 | 1928673 | n/a（对照） |
| KnowDesk-Plain (knowdesk_plain + 自研混合检索 + Fail-Closed Guard) | 0.1667 | 0.8832 | 0.6802 | 0.8928 | 1112684 | [-0.1333, 0.2000] INCONCLUSIVE |
| KnowDesk-LangGraph (knowdesk_graph + 自研混合检索 + Fail-Closed Guard) | 0.1667 | 0.8957 | 0.6396 | 0.9017 | 995611 | [-0.1000, 0.1667] INCONCLUSIVE |

## knowdesk_graph + hybrid 失败归因

| 类别 | 任务数 |
|---|---:|
| 0. 通过 | 5 |
| 3. 隐藏工具未发现或未解锁 | 3 |
| 4. 漏调关键业务工具 | 2 |
| 5. 工具参数填写错误 | 17 |
| 6. 子任务顺序或条款推理 | 3 |
