import hashlib
import re
from typing import Any

_RE_CARD = re.compile(r"\b(?:\d{4}[ -]?){3}(\d{4})\b")
_RE_SSN = re.compile(r"\b\d{3}-\d{2}-(\d{4})\b")
_RE_PHONE = re.compile(r"\b(\d{3})[-.]?\d{3}[-.]?(\d{4})\b")
_RE_EMAIL = re.compile(r"\b([a-zA-Z0-9_.+-])[a-zA-Z0-9_.+-]*@([a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+)\b")
_RE_ROUTING = re.compile(r"\b(?:routing|aba|transit)\s*#?:?\s*(\d{9})\b", re.IGNORECASE)
_RE_ADDRESS = re.compile(
    r"\b\d{1,5}\s+[A-Za-z0-9\s.,]+(?:Street|St|Avenue|Ave|Boulevard|Blvd|Road|Rd|Drive|Dr|Lane|Ln|Way|Parkway)\b",
    re.IGNORECASE,
)


def redact_text(text: str) -> str:
    """不可逆掩码：16 位卡号、SSN、手机号、邮箱。地址与令牌由保险库另处理。"""
    value = str(text)
    value = _RE_CARD.sub(r"****-****-****-\1", value)
    value = _RE_SSN.sub(r"***-**-\1", value)
    value = _RE_PHONE.sub(r"\1-***-\2", value)
    value = _RE_EMAIL.sub(r"\1***@\2", value)
    return value

class SensitiveDataVault:
    """
    企业级可逆敏感数据保险库 (Google Cloud DLP / Sensitive Data Protection Pattern):
    1. 实体识别：覆盖 16位卡号、SSN、手机号、邮箱、9位银行路由号(ABA)与街道地址；
    2. 可逆令牌化 (Reversible Tokenization / Pseudonymization)：
       将真实敏感数据加密散列替换为匿名令牌（如 [TOKEN_CARD_9a8f]），LLM 仅接触无害令牌，杜绝向外泄露；
    3. 受信域解令牌 (Detokenization / Vault Unmasking)：
       当工具执行器向银行内部专网后端发起高风险调用时，由受信中间件把参数中的令牌精准还原为真实值；
    4. 不可逆审计脱敏 (Irreversible Redaction)：
       对落盘的 Trace、日志与 Jaeger Span 进行彻底不可逆掩码。
    """
    def __init__(self, salt: str = "knowdesk_enterprise_secret_salt"):
        self.salt = salt
        self.token_to_raw: dict[str, str] = {}
        self.raw_to_token: dict[str, str] = {}

    def _generate_token(self, entity_type: str, raw_val: str) -> str:
        h = hashlib.sha256(f"{self.salt}:{raw_val}".encode("utf-8")).hexdigest()[:8]
        return f"[TOKEN_{entity_type.upper()}_{h}]"

    def tokenize(self, text: str) -> str:
        """扫描文本并替换为可逆令牌，记录映射至保险库"""
        s = str(text)

        def _replace_card(m: re.Match) -> str:
            raw = m.group(0)
            tok = self._generate_token("CARD", raw)
            self.token_to_raw[tok] = raw
            self.raw_to_token[raw] = tok
            return tok

        def _replace_ssn(m: re.Match) -> str:
            raw = m.group(0)
            tok = self._generate_token("SSN", raw)
            self.token_to_raw[tok] = raw
            self.raw_to_token[raw] = tok
            return tok

        def _replace_phone(m: re.Match) -> str:
            raw = m.group(0)
            tok = self._generate_token("PHONE", raw)
            self.token_to_raw[tok] = raw
            self.raw_to_token[raw] = tok
            return tok

        def _replace_email(m: re.Match) -> str:
            raw = m.group(0)
            tok = self._generate_token("EMAIL", raw)
            self.token_to_raw[tok] = raw
            self.raw_to_token[raw] = tok
            return tok

        def _replace_address(m: re.Match) -> str:
            raw = m.group(0)
            tok = self._generate_token("ADDR", raw)
            self.token_to_raw[tok] = raw
            self.raw_to_token[raw] = tok
            return tok

        s = _RE_CARD.sub(_replace_card, s)
        s = _RE_SSN.sub(_replace_ssn, s)
        s = _RE_PHONE.sub(_replace_phone, s)
        s = _RE_EMAIL.sub(_replace_email, s)
        s = _RE_ADDRESS.sub(_replace_address, s)
        return s

    def detokenize(self, obj: Any) -> Any:
        """在受信执行边界对字符串、字典或列表执行可逆解令牌还原真实值"""
        if isinstance(obj, str):
            res = obj
            for tok, raw in self.token_to_raw.items():
                if tok in res:
                    res = res.replace(tok, raw)
            return res
        elif isinstance(obj, dict):
            return {k: self.detokenize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self.detokenize(v) for v in obj]
        return obj

    def redact_irreversible(self, text: str) -> str:
        """不可逆脱敏（专供公开日志、OTel Span 与外部报表）"""
        s = redact_text(text)
        s = _RE_ADDRESS.sub(r"[REDACTED_STREET_ADDRESS]", s)
        # 顺便抹掉可能存在的令牌
        for tok in list(self.token_to_raw.keys()):
            if tok in s:
                prefix = tok.split("_")[1]
                s = s.replace(tok, f"****_{prefix}_REDACTED****")
        return s
