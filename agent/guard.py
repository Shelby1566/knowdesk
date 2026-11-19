import base64
import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml
from tools.schemas import StructuredToolError, is_tool_result_success

DEFAULT_RULES_PATH = Path(__file__).resolve().parent / "guard_rules.yaml"

STRICT_VERIFICATION_TOOLS = frozenset({
    "get_user_details",
    "get_user_information_by_id",
    "get_user_information_by_name",
    "get_user_information_by_email",
    "get_account_details",
    "get_credit_card_details",
    "get_credit_card_accounts_by_user",
    "lookup_user",
    "verify_user",
    "authenticate_user",
    "log_verification",
})

_ZERO_WIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060\ufeff]")
_B64_CANDIDATE_RE = re.compile(r"[A-Za-z0-9+/]{16,}={0,2}")

def normalize_for_security_inspection(raw_text: str) -> str:
    """
    纵深防御归一化层 (Defense-in-Depth Normalization)：
    1. 剥离零宽混淆字符；
    2. Unicode NFKD 规范化（将全角字符如 ｉｇｎｏｒｅ 还原为标准 ASCII ignore）；
    3. URL 解码；
    4. 自动扫描并解码潜在的 Base64 编码载荷拼回检测文本。
    """
    s = _ZERO_WIDTH_RE.sub("", raw_text)
    s = unicodedata.normalize("NFKD", s)
    s = urllib.parse.unquote(s)
    decoded_extras: list[str] = []
    for match in _B64_CANDIDATE_RE.findall(s):
        if len(match) % 4 == 0:
            try:
                dec = base64.b64decode(match, validate=True).decode("utf-8", errors="ignore")
                if dec and any(c.isalpha() for c in dec):
                    decoded_extras.append(dec)
            except Exception:
                pass
    if decoded_extras:
        s = s + " " + " ".join(decoded_extras)
    return s.lower()

@dataclass
class RuleVerdict:
    passed: bool
    risk_category: Literal["clean", "jailbreak", "data_leakage", "toxic_financial_promise", "system_override"]
    confidence: float
    reason: str = ""


class LocalRuleChecker:
    """本地输入/输出规则检查。正则直接跑在进程内，不调用外部安全服务。"""

    INPUT_JAILBREAKS = [
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
        re.compile(r"\b(?:\d{4}[ -]?){3}\d{4}\b"),
        re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    ]
    EXFILTRATION_RE = re.compile(r"!\[.*?\]\((https?://[^\s)]+)\)")

    def inspect_input(self, text: str) -> RuleVerdict:
        sample = str(text)
        for pattern in self.INPUT_JAILBREAKS:
            if pattern.search(sample):
                return RuleVerdict(
                    passed=False,
                    risk_category="jailbreak",
                    confidence=0.98,
                    reason=f"Local rule checker blocked a jailbreak pattern: '{pattern.pattern}'",
                )
        return RuleVerdict(passed=True, risk_category="clean", confidence=1.0)

    def inspect_output(self, response_text: str) -> RuleVerdict:
        sample = str(response_text)
        for pattern in self.DATA_LEAKAGE_RE:
            if pattern.search(sample):
                return RuleVerdict(
                    passed=False,
                    risk_category="data_leakage",
                    confidence=0.99,
                    reason="Local rule checker blocked raw payment or identity data in the response.",
                )
        for pattern in self.FINANCIAL_PROHIBITED_CLAIMS:
            if pattern.search(sample):
                return RuleVerdict(
                    passed=False,
                    risk_category="toxic_financial_promise",
                    confidence=0.95,
                    reason="Local rule checker blocked an absolute financial guarantee in the response.",
                )
        if self.EXFILTRATION_RE.search(sample):
            return RuleVerdict(
                passed=False,
                risk_category="jailbreak",
                confidence=0.92,
                reason="Local rule checker blocked a markdown image exfiltration URL in the response.",
            )
        return RuleVerdict(passed=True, risk_category="clean", confidence=1.0)


@dataclass
class UnlockedToolSchema:
    tool_name: str
    description: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    optional_params: list[str] = field(default_factory=list)
    enums: dict[str, list[Any]] = field(default_factory=dict)


def _iter_json_dicts(raw: str):
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        parsed = None
    if isinstance(parsed, dict):
        yield parsed
        return
    decoder = json.JSONDecoder()
    index = 0
    while index < len(raw):
        brace = raw.find("{", index)
        if brace < 0:
            return
        try:
            obj, end = decoder.raw_decode(raw[brace:])
        except json.JSONDecodeError:
            index = brace + 1
            continue
        if isinstance(obj, dict):
            yield obj
        index = brace + max(end, 1)


def _schema_from_json_object(obj: dict[str, Any], tool_name: str) -> UnlockedToolSchema | None:
    node = obj if isinstance(obj.get("properties"), dict) else None
    parameters = obj.get("parameters")
    if node is None and isinstance(parameters, dict) and isinstance(parameters.get("properties"), dict):
        node = parameters
    if node is None:
        return None
    properties = node.get("properties") or {}
    required = [str(name) for name in (node.get("required") or [])]
    params: dict[str, str] = {}
    enums: dict[str, list[Any]] = {}
    optional: list[str] = []
    for pname, spec in properties.items():
        key = str(pname)
        if isinstance(spec, dict):
            params[key] = str(spec.get("type", "string")).lower()
            if isinstance(spec.get("enum"), list):
                enums[key] = list(spec["enum"])
        else:
            params[key] = "string"
        if key not in required:
            optional.append(key)
    name = tool_name or str(obj.get("name") or obj.get("title") or "")
    if not name:
        return None
    return UnlockedToolSchema(
        tool_name=name,
        description=str(obj.get("description") or ""),
        parameters=params,
        required_params=required,
        optional_params=optional,
        enums=enums,
    )


def _value_matches_type(expected: str, value: Any) -> bool:
    kind = expected.lower().strip()
    if kind in {"string", "str"}:
        return isinstance(value, str)
    if kind in {"integer", "int"}:
        return isinstance(value, int) and not isinstance(value, bool)
    if kind in {"number", "float"}:
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if kind in {"boolean", "bool"}:
        return isinstance(value, bool)
    if kind in {"object", "dict"}:
        return isinstance(value, dict)
    if kind in {"array", "list"}:
        return isinstance(value, list)
    return True

def parse_unlocked_tool_schema(content: str, default_tool_name: str = "") -> UnlockedToolSchema | None:
    """
    动态从环境返回的 ToolMessage 或文档中提取解锁工具的参数 Schema：
    - 优先解析标准 JSON Schema（required、properties.type、properties.enum，可嵌在 parameters 下）
    - 兼容 Markdown/YAML 列表 (Parameters:\n - param: type (required|optional))
    - 兼容 Python 签名 (func(a: str, b: int))
    """
    if not content or not str(content).strip():
        return None
    raw = str(content)

    tool_m = re.search(r"Tool (?:unlocked:\s*|:\s*)([a-zA-Z0-9_]+)", raw, re.IGNORECASE)
    tool_name = tool_m.group(1) if tool_m else default_tool_name
    for obj in _iter_json_dicts(raw):
        schema = _schema_from_json_object(obj, tool_name)
        if schema is not None:
            return schema

    params: dict[str, str] = {}
    req: list[str] = []
    opt: list[str] = []

    # 2. 匹配 Markdown/YAML 参数列表
    param_pattern = re.compile(
        r"-\s*([a-zA-Z0-9_]+)\s*:\s*([a-zA-Z0-9_]+)\s*\((required|optional)\)",
        re.IGNORECASE,
    )
    for m in param_pattern.finditer(raw):
        pname = m.group(1)
        ptype = m.group(2).lower()
        req_opt = m.group(3).lower()
        params[pname] = ptype
        if req_opt == "required":
            req.append(pname)
        else:
            opt.append(pname)

    # 3. 匹配函数签名格式 tool_name(param1: type, ...)
    if not params:
        sig_m = re.search(r"\b([a-zA-Z0-9_]+)\s*\(([^)]+)\)", raw)
        if sig_m:
            if not tool_name:
                tool_name = sig_m.group(1)
            raw_args = sig_m.group(2)
            for part in raw_args.split(","):
                part = part.strip()
                if ":" in part:
                    arg_name, arg_type = part.split(":", 1)
                    arg_name = arg_name.strip()
                    arg_type = arg_type.strip().split("=")[0].strip()
                    if arg_name:
                        params[arg_name] = arg_type
                        if "=" in part:
                            opt.append(arg_name)
                        else:
                            req.append(arg_name)

    if not tool_name:
        return None
    return UnlockedToolSchema(
        tool_name=tool_name,
        parameters=params,
        required_params=req,
        optional_params=opt,
    )

@dataclass
class SecurityState:
    verified: bool = False
    unlocked_tools: set[str] = field(default_factory=set)
    unlocked_schemas: dict[str, UnlockedToolSchema] = field(default_factory=dict)
    approved_idempotency_keys: set[str] = field(default_factory=set)

@dataclass(frozen=True)
class GuardDecision:
    action: Literal["allow", "block", "require_approval"]
    real_tool_name: str
    real_args: dict[str, Any]
    rule_id: str | None = None
    reason: str = ""
    error_payload: StructuredToolError | None = None

def unwrap_tool_call(tool_name: str, raw_args: dict[str, Any] | str) -> tuple[str, dict[str, Any], str | None]:
    if isinstance(raw_args, str):
        try:
            args_dict = json.loads(raw_args) if raw_args.strip() else {}
        except json.JSONDecodeError as e:
            return tool_name, {}, f"R01: outer arguments is not valid JSON: {e}"
        if not isinstance(args_dict, dict):
            return tool_name, {}, "R01: outer arguments JSON must decode to an object."
    elif isinstance(raw_args, dict):
        args_dict = dict(raw_args)
    else:
        return tool_name, {}, "R01: outer arguments must be a JSON object."

    if tool_name == "call_discoverable_agent_tool":
        inner_name = str(args_dict.get("agent_tool_name", "")).strip()
        if not inner_name:
            return tool_name, args_dict, "Missing required 'agent_tool_name'."
        inner_raw = args_dict.get("arguments", "{}")
        if isinstance(inner_raw, str):
            try:
                inner_args = json.loads(inner_raw) if inner_raw.strip() else {}
            except json.JSONDecodeError as e:
                return inner_name, {}, f"Inner 'arguments' for '{inner_name}' is not valid JSON: {e}"
        elif isinstance(inner_raw, dict):
            inner_args = dict(inner_raw)
        else:
            return inner_name, {}, "Inner 'arguments' must be a JSON string or object."
        if not isinstance(inner_args, dict):
            return inner_name, {}, "Inner 'arguments' must decode to a JSON object."
        return inner_name, inner_args, None

    if tool_name == "give_discoverable_user_tool":
        inner_name = str(args_dict.get("discoverable_tool_name", "")).strip()
        return (inner_name or tool_name), args_dict, None

    return tool_name, args_dict, None

class GuardEngine:
    def __init__(
        self,
        rules_path: Path = DEFAULT_RULES_PATH,
        enforce_unlock: bool = True,
        enforce_verification: bool = True,
        require_human_confirm_on_write: bool = False,
    ):
        self.config = yaml.safe_load(rules_path.read_text(encoding="utf-8"))
        self.rules = {r["id"]: r for r in self.config.get("rules", [])}
        self.enforce_unlock = enforce_unlock
        self.enforce_verification = enforce_verification
        self.require_human_confirm_on_write = require_human_confirm_on_write
        self.blocked_substrings = [
            s.lower() for s in self.rules.get("R02_PROMPT_INJECTION_PAYLOAD", {}).get("blocked_substrings", [])
        ]
        r04 = self.rules.get("R04_HIGH_RISK_WRITE_VERIFICATION_AND_APPROVAL", {})
        self.high_risk_prefixes = tuple(r04.get("high_risk_prefixes", []))
        self.read_or_verify_tools = set(r04.get("read_or_verify_tools", []))
        self.local_rules = LocalRuleChecker()

    def inspect_output(self, response_text: str) -> RuleVerdict:
        """本地输出规则：拦截明文卡号/身份号、绝对金融承诺和 Markdown 外带。"""
        return self.local_rules.inspect_output(response_text)

    def is_high_risk_write(self, tool_name: str, real_tool_name: str) -> bool:
        if real_tool_name in self.read_or_verify_tools or real_tool_name in STRICT_VERIFICATION_TOOLS:
            return False
        if real_tool_name.startswith(("get_", "list_", "search_", "check_", "view_", "calculate_", "verify_")):
            return False
        if real_tool_name.startswith(self.high_risk_prefixes) or tool_name == "call_discoverable_agent_tool":
            return True
        return False

    def inspect(
        self, tool_name: str, raw_args: dict[str, Any] | str, sec_state: SecurityState, idem_key: str | None = None
    ) -> GuardDecision:
        real_name, real_args, parse_err = unwrap_tool_call(tool_name, raw_args)
        if parse_err is not None:
            err = StructuredToolError(
                error_type="invalid_arguments", message=parse_err, rule_id="R01_UNWRAP_MALFORMED_DISPATCH"
            )
            return GuardDecision("block", real_name, real_args, "R01_UNWRAP_MALFORMED_DISPATCH", parse_err, err)

        normalized_payload = normalize_for_security_inspection(json.dumps(real_args, ensure_ascii=False))
        local_in = self.local_rules.inspect_input(normalized_payload)
        if not local_in.passed:
            err = StructuredToolError(
                error_type="guard_blocked", message=local_in.reason, rule_id="R02_PROMPT_INJECTION_PAYLOAD"
            )
            return GuardDecision("block", real_name, real_args, "R02_PROMPT_INJECTION_PAYLOAD", local_in.reason, err)

        for bad in self.blocked_substrings:
            if bad in normalized_payload:
                msg = f"Blocked prompt injection payload matching '{bad}' (after NFKD/Base64/URL normalization)."
                err = StructuredToolError(
                    error_type="guard_blocked", message=msg, rule_id="R02_PROMPT_INJECTION_PAYLOAD"
                )
                return GuardDecision("block", real_name, real_args, "R02_PROMPT_INJECTION_PAYLOAD", msg, err)

        if (
            self.enforce_unlock
            and tool_name == "call_discoverable_agent_tool"
            and real_name not in sec_state.unlocked_tools
        ):
            msg = f"Discoverable tool '{real_name}' must be unlocked first."
            err = StructuredToolError(
                error_type="guard_blocked", message=msg, rule_id="R03_REQUIRE_UNLOCK_BEFORE_DISCOVERABLE_CALL"
            )
            return GuardDecision("block", real_name, real_args, "R03_REQUIRE_UNLOCK_BEFORE_DISCOVERABLE_CALL", msg, err)

        schema = sec_state.unlocked_schemas.get(real_name)
        if schema is not None:
            missing = [
                p for p in schema.required_params
                if p not in real_args or real_args[p] is None or (isinstance(real_args[p], str) and not str(real_args[p]).strip())
            ]
            if missing:
                msg = (
                    f"Missing required argument(s) {missing} for unlocked tool '{real_name}'. "
                    f"Required parameters: {schema.required_params}"
                )
                err = StructuredToolError(
                    error_type="invalid_arguments",
                    message=msg,
                    rule_id="R05_DYNAMIC_SCHEMA_VALIDATION",
                    retryable=True,
                )
                return GuardDecision("block", real_name, real_args, "R05_DYNAMIC_SCHEMA_VALIDATION", msg, err)
            for pname, expected in schema.parameters.items():
                if pname not in real_args or real_args[pname] is None:
                    continue
                value = real_args[pname]
                if not _value_matches_type(expected, value):
                    msg = (
                        f"Argument '{pname}' for unlocked tool '{real_name}' has invalid type "
                        f"'{type(value).__name__}', expected {expected}."
                    )
                    err = StructuredToolError(
                        error_type="invalid_arguments",
                        message=msg,
                        rule_id="R05_DYNAMIC_SCHEMA_VALIDATION",
                        retryable=True,
                    )
                    return GuardDecision("block", real_name, real_args, "R05_DYNAMIC_SCHEMA_VALIDATION", msg, err)
                allowed = schema.enums.get(pname)
                if allowed is not None and value not in allowed:
                    msg = (
                        f"Argument '{pname}' for unlocked tool '{real_name}' has illegal enum value {value!r}. "
                        f"Allowed: {allowed}."
                    )
                    err = StructuredToolError(
                        error_type="invalid_arguments",
                        message=msg,
                        rule_id="R05_DYNAMIC_SCHEMA_VALIDATION",
                        retryable=True,
                    )
                    return GuardDecision("block", real_name, real_args, "R05_DYNAMIC_SCHEMA_VALIDATION", msg, err)

        if self.is_high_risk_write(tool_name, real_name):
            if self.enforce_verification and not sec_state.verified:
                msg = f"High-risk operation '{real_name}' requires prior verification."
                err = StructuredToolError(
                    error_type="unauthorized", message=msg, rule_id="R04_HIGH_RISK_WRITE_VERIFICATION_AND_APPROVAL"
                )
                return GuardDecision("block", real_name, real_args, "R04_HIGH_RISK_WRITE_VERIFICATION_AND_APPROVAL", msg, err)

            if self.require_human_confirm_on_write and (
                idem_key is None or idem_key not in sec_state.approved_idempotency_keys
            ):
                return GuardDecision(
                    "require_approval", real_name, real_args, "R04_HIGH_RISK_WRITE_VERIFICATION_AND_APPROVAL",
                    f"Write operation '{real_name}' requires human approval."
                )

        return GuardDecision("allow", real_name, real_args)

    def update_state_after_success(
        self,
        tool_name: str,
        raw_args: dict[str, Any] | str,
        sec_state: SecurityState,
        result_content: str | None = None,
    ) -> None:
        if result_content is not None and not is_tool_result_success(result_content):
            return
        real_name, real_args, _ = unwrap_tool_call(tool_name, raw_args)
        if tool_name == "unlock_discoverable_agent_tool":
            unlocked = str(
                real_args.get("agent_tool_name")
                or (raw_args.get("agent_tool_name", "") if isinstance(raw_args, dict) else "")
            ).strip()
            if unlocked:
                sec_state.unlocked_tools.add(unlocked)
            if result_content:
                schema = parse_unlocked_tool_schema(result_content, default_tool_name=unlocked)
                if schema and schema.tool_name:
                    sec_state.unlocked_schemas[schema.tool_name] = schema
                    sec_state.unlocked_tools.add(schema.tool_name)
        elif result_content and (
            "Tool unlocked:" in result_content or "Parameters:" in result_content or '"properties"' in result_content
        ):
            schema = parse_unlocked_tool_schema(result_content)
            if schema and schema.tool_name:
                sec_state.unlocked_schemas[schema.tool_name] = schema
                sec_state.unlocked_tools.add(schema.tool_name)

        if real_name in STRICT_VERIFICATION_TOOLS or real_name.startswith(("verify_", "authenticate_")):
            sec_state.verified = True
