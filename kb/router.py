from kb.loader import Document

# 将 71 个子类前缀归并为 11 个银行产品线大类（与第 2.3 节一致）
PRODUCT_LINES = (
    "credit_card",
    "checking_account",
    "savings_account",
    "mortgage_loan",
    "auto_loan",
    "personal_loan",
    "wire_transfer",
    "fraud_dispute",
    "online_banking",
    "business_banking",
    "general_policy",
)

def map_doc_to_product_line(doc_id: str, title: str = "") -> str:
    s = f"{doc_id} {title}".lower()
    if any(k in s for k in ("credit_card", "card", "apr", "cash_back", "rewards", "platinum", "sapphire")):
        return "credit_card"
    if any(k in s for k in ("checking", "overdraft", "debit")):
        return "checking_account"
    if any(k in s for k in ("saving", "cd_", "certificate", "interest_rate", "money_market")):
        return "savings_account"
    if any(k in s for k in ("mortgage", "home_equity", "heloc", "escrow", "refinance")):
        return "mortgage_loan"
    if any(k in s for k in ("auto", "vehicle", "car_loan")):
        return "auto_loan"
    if any(k in s for k in ("personal_loan", "student_loan", "installment", "loan")):
        return "personal_loan"
    if any(k in s for k in ("wire", "ach", "swift", "remittance", "transfer", "zelle")):
        return "wire_transfer"
    if any(k in s for k in ("fraud", "dispute", "chargeback", "unauthorized", "scam", "stolen")):
        return "fraud_dispute"
    if any(k in s for k in ("online", "mobile", "app", "password", "mfa", "login", "2fa")):
        return "online_banking"
    if any(k in s for k in ("business", "commercial", "merchant", "payroll")):
        return "business_banking"
    return "general_policy"

def predict_product_lines_keyword(query: str) -> set[str]:
    """零延迟关键词+规则多标签路由基线"""
    q = query.lower()
    predicted: set[str] = set()
    for line in PRODUCT_LINES[:-1]:
        # 对每个产品线检查关键词命中
        if map_doc_to_product_line("", q) == line:
            predicted.add(line)
    # 补充多标签独立检测
    if any(k in q for k in ("credit card", "cash back", "annual fee", "foreign transaction", "card")):
        predicted.add("credit_card")
    if any(k in q for k in ("checking", "overdraft", "debit")):
        predicted.add("checking_account")
    if any(k in q for k in ("saving", "interest", "apy", "cd")):
        predicted.add("savings_account")
    if any(k in q for k in ("mortgage", "home", "refinance", "escrow")):
        predicted.add("mortgage_loan")
    if any(k in q for k in ("wire", "transfer", "ach", "international", "zelle")):
        predicted.add("wire_transfer")
    if any(k in q for k in ("fraud", "dispute", "unauthorized", "stolen", "compromised")):
        predicted.add("fraud_dispute")
    return predicted

def apply_intent_boost(
    ranked_docs: list[tuple[str, float]],
    doc_to_line: dict[str, str],
    predicted_lines: set[str],
    beta: float = 0.15,
    k: int = 10,
) -> list[tuple[str, float]]:
    """M8 软加权（绝不硬过滤）：s'(d) = s(d) * (1 + beta * I[line(d) in Y_hat])"""
    if not predicted_lines or beta == 0.0:
        return ranked_docs[:k]
    boosted = [
        (did, sc * (1.0 + beta) if doc_to_line.get(did) in predicted_lines else sc)
        for did, sc in ranked_docs
    ]
    boosted.sort(key=lambda x: x[1], reverse=True)
    return boosted[:k]

def evaluate_multilabel_f1(
    y_true_list: list[set[str]],
    y_pred_list: list[set[str]],
    labels: tuple[str, ...] = PRODUCT_LINES,
) -> dict[str, float]:
    """计算多标签 Micro-P/R/F1 与 Macro-F1（验收 ③）"""
    tp_total = fp_total = fn_total = 0
    f1_per_class: list[float] = []

    for c in labels:
        tp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if c in yt and c in yp)
        fp = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if c not in yt and c in yp)
        fn = sum(1 for yt, yp in zip(y_true_list, y_pred_list) if c in yt and c not in yp)
        tp_total += tp
        fp_total += fp
        fn_total += fn
        denom = 2 * tp + fp + fn
        if any(c in yt for yt in y_true_list):
            f1_per_class.append((2 * tp / denom) if denom > 0 else 0.0)

    micro_p = tp_total / (tp_total + fp_total) if (tp_total + fp_total) > 0 else 0.0
    micro_r = tp_total / (tp_total + fn_total) if (tp_total + fn_total) > 0 else 0.0
    micro_f1 = (2 * micro_p * micro_r / (micro_p + micro_r)) if (micro_p + micro_r) > 0 else 0.0
    macro_f1 = sum(f1_per_class) / len(f1_per_class) if f1_per_class else 0.0
    return {
        "micro_precision": micro_p,
        "micro_recall": micro_r,
        "micro_f1": micro_f1,
        "macro_f1": macro_f1,
    }
