import base64
import json
import re
import unicodedata
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
import yaml
from tools.schemas import StructuredToolError

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
class UnlockedToolSchema:
    tool_name: str
    description: str = ""
    parameters: dict[str, str] = field(default_factory=dict)
    required_params: list[str] = field(default_factory=list)
    optional_params: list[str] = field(default_factory=list)

def parse_unlocked_tool_schema(content: str, default_tool_name: str = "") -> UnlockedToolSchema | None:
    """
    动态从环境返回的 ToolMessage 或文档中提取解锁工具的参数 Schema：
    - 支持 Markdown/YAML 列表格式 (Parameters:\n - param: type (required|optional))
    - 支持 Python 签名格式 (func(a: str, b: int))
    - 支持 JSON Schema 格式 ({"properties": ..., "required": ...})
    """
    if not content or not str(content).strip():
        return None
    raw = str(content)

    # 1. 提取工具名
    tool_m = re.search(r"Tool (?:unlocked:\s*|:\s*)([a-zA-Z0-9_]+)", raw, re.IGNORECASE)
    tool_name = tool_m.group(1) if tool_m else default_tool_name

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
            return tool_name, {}, f"Outer arguments is not valid JSON: {e}"
    elif isinstance(raw_args, dict):
        args_dict = dict(raw_args)
    else:
        return tool_name, {}, "Outer arguments must be a JSON object."

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

        # R05: 动态 Schema 感知参数拦截（运行时前置拦截缺失必填参数）
        if (
            tool_name == "call_discoverable_agent_tool"
            and real_name in sec_state.unlocked_schemas
        ):
            schema = sec_state.unlocked_schemas[real_name]
            missing = [
                p for p in schema.required_params
                if p not in real_args or real_args[p] is None or (isinstance(real_args[p], str) and not real_args[p].strip())
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
        elif result_content and ("Tool unlocked:" in result_content or "Parameters:" in result_content):
            schema = parse_unlocked_tool_schema(result_content)
            if schema and schema.tool_name:
                sec_state.unlocked_schemas[schema.tool_name] = schema
                sec_state.unlocked_tools.add(schema.tool_name)

        if real_name in STRICT_VERIFICATION_TOOLS or real_name.startswith(("verify_", "authenticate_")):
            sec_state.verified = True
