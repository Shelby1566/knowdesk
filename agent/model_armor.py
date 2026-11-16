import re
from dataclasses import dataclass
from typing import Literal

@dataclass
class ArmorVerdict:
    passed: bool
    risk_category: Literal["clean", "jailbreak", "data_leakage", "toxic_financial_promise", "system_override"]
    confidence: float
    reason: str = ""

class VertexModelArmor:
    """
    企业级双向模型装甲 (Google Cloud Vertex AI Model Armor / ShieldGemma Pattern):
    1. Input Armor (入口装甲):
       - 拦截多语言越狱与指令覆盖 (English, Spanish, Chinese, Russian, etc.)
       - 拦截角色扮演 (DAN, Developer Mode, Persona Hijack) 与系统提示探测
    2. Output Armor (出口装甲):
       - 拦截响应中的敏感信息明文泄露 (Raw Card / SSN)
       - 拦截未经验证的绝对金融承诺 (如 100% 承诺、保证收益、免除一切风控)
       - 拦截 Markdown 图片/超链接外发外传漏洞 (Exfiltration Injection)
    """
    MULTILINGUAL_JAILBREAKS = [
        re.compile(r"\b(?:ignore|forget|disregard)\s+(?:all\s+)?(?:previous|prior|above)\s+(?:instructions|rules|prompts)\b", re.IGNORECASE),
        re.compile(r"\b(?:ignora|olvida)\s+(?:todas\s+(?:las\s+)?|las\s+)?instrucciones\s+(?:anteriores|previas)\b", re.IGNORECASE),
        re.compile(r"(?:忽略|无视|丢弃|绕过).*?(?:指令|提示词|规则|约束|开发者模式)", re.IGNORECASE),
        re.compile(r"\b(?:забудь|игнорируй)\s+(?:все\s+)?предыдущие\s+инструкции\b", re.IGNORECASE),
        re.compile(r"\b(?:dan|developer\s+mode|jailbreak|evil\s*confidant|unfiltered\s+mode)\b", re.IGNORECASE),
        re.compile(r"\byou\s+are\s+now\s+(?:in\s+)?(?:god\s+mode|unrestricted|an\s+ai\s+without\s+rules)\b", re.IGNORECASE),
    ]

    FINANCIAL_PROHIBITED_CLAIMS = [
        re.compile(r"\bguarantee(?:d)?\s+100%\s+(?:return|profit|refund\s+without\s+condition)\b", re.IGNORECASE),
        re.compile(r"(?:保证|承诺)(?:百分之百|100%)(?:全额退款无需审核|绝对无风险|超高收益)", re.IGNORECASE),
        re.compile(r"\bwaive\s+all\s+(?:regulatory|fbi|irs|compliance)\s+requirements\b", re.IGNORECASE),
    ]

    DATA_LEAKAGE_RE = [
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),  # 16-digit card unmasked
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),       # SSN unmasked
    ]

    EXFILTRATION_RE = re.compile(r"!\[.*?\]\((https?://[^\s)]+)\)")

    def inspect_input(self, text: str) -> ArmorVerdict:
        """检查输入提示词与用户意图"""
        s = str(text)
        for pattern in self.MULTILINGUAL_JAILBREAKS:
            if pattern.search(s):
                return ArmorVerdict(
                    passed=False,
                    risk_category="jailbreak",
                    confidence=0.98,
                    reason=f"Model Armor detected jailbreak pattern: '{pattern.pattern}'",
                )
        return ArmorVerdict(passed=True, risk_category="clean", confidence=1.0)

    def inspect_output(self, response_text: str) -> ArmorVerdict:
        """检查模型生成回复是否存在泄露、虚假违规金融承诺或外联窃密"""
        s = str(response_text)
        # 1. 检查数据泄露
        for leak_p in self.DATA_LEAKAGE_RE:
            if leak_p.search(s):
                return ArmorVerdict(
                    passed=False,
                    risk_category="data_leakage",
                    confidence=0.99,
                    reason="Model Armor blocked response containing raw unmasked payment/PII data.",
                )

        # 2. 检查违规绝对金融承诺
        for fin_p in self.FINANCIAL_PROHIBITED_CLAIMS:
            if fin_p.search(s):
                return ArmorVerdict(
                    passed=False,
                    risk_category="toxic_financial_promise",
                    confidence=0.95,
                    reason="Model Armor blocked unbacked absolute financial guarantee in agent output.",
                )

        # 3. 检查外联外带
        if self.EXFILTRATION_RE.search(s):
            return ArmorVerdict(
                passed=False,
                risk_category="jailbreak",
                confidence=0.92,
                reason="Model Armor blocked Markdown image exfiltration URL in agent response.",
            )

        return ArmorVerdict(passed=True, risk_category="clean", confidence=1.0)
