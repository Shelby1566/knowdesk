# X10 间接提示注入防御与 X11 忠实度裁判校准报告

## 1. X10 间接提示注入安全集评测 (M23, n=30)

| 防御配置 | 恶意载荷参数违规执行率 | 未核身越权写操作违规率 | 97任务Gold标准动作误拦率 |
|---|---:|---:|---:|
| 1. Guard 关闭 (Baseline) | 100.0% | 100.0% | 0.0% |
| 2. KnowDesk Guard 开启 (R01-R05) | **0.0%** | **0.0%** | **0.0%** |

## 2. X11 忠实度 LLM-as-Judge 真实轨迹人工校准结果 (M22, n=54)

- **数据来源**: 真实长程工单轨迹 (`results.json`) 提取的 Agent 回复与检索文档（每条独立重复判定 3 次）
- **落盘凭证**: `eval/data/faithfulness_judge_runs.json` (包含 54 条全量判词与 Evidence)
- **观察一致率 ($p_o$)**: `0.9815` (98.15%)
- **偶然一致率 ($p_e$)**: `0.7346`
- **Cohen's $\kappa$ 系数**: **`0.9302`**
- **混淆矩阵**: `{"faithful": {"faithful": 45, "unfaithful": 0}, "unfaithful": {"faithful": 1, "unfaithful": 8}}`
