"""银行工单的显式阶段。模型只看到当前阶段的短指令，阶段由已发生的工具调用决定。"""

from knowdesk.agent.guard import SecurityState

RETRIEVAL_TOOLS = frozenset({"KB_search", "KB_search_bm25", "KB_search_dense", "grep"})
DISCOVERABLE_TOOLS = frozenset({"unlock_discoverable_agent_tool", "call_discoverable_agent_tool"})


def _tool_names(messages: list) -> list[str]:
    names: list[str] = []
    for message in messages:
        calls = getattr(message, "tool_calls", None)
        if calls is None and isinstance(message, dict):
            calls = message.get("tool_calls") or []
        for call in calls or []:
            if isinstance(call, dict):
                names.append(str(call.get("name", "")))
            else:
                names.append(str(getattr(call, "name", "")))
    return names


def _schema_lines(sec: SecurityState) -> list[str]:
    lines: list[str] = []
    for name, schema in (sec.unlocked_schemas or {}).items():
        required = getattr(schema, "required_params", None)
        if required is None and isinstance(schema, dict):
            required = schema.get("required_params", [])
        if not isinstance(required, list):
            required = []
        shown = ", ".join(str(item) for item in required) or "(see the unlock result)"
        lines.append(f"- {name}: {shown}")
    return lines


def stage_directive(
    messages: list,
    sec: SecurityState,
    available_tools: set[str] | None = None,
) -> str:
    """返回当前阶段的一句操作指令。无银行工具时返回空字符串。"""
    available = set(available_tools or [])
    banking = bool(available & (RETRIEVAL_TOOLS | DISCOVERABLE_TOOLS))
    if not banking:
        return ""

    called = _tool_names(messages)
    can_retrieve = bool(available & RETRIEVAL_TOOLS)
    retrieved = (not can_retrieve) or any(name in RETRIEVAL_TOOLS for name in called)
    verified = bool(sec.verified) or ("log_verification" in called)
    unlocked = sorted(sec.unlocked_tools)
    uses_discoverable = bool(available & DISCOVERABLE_TOOLS)
    called_inner = "call_discoverable_agent_tool" in called

    if not retrieved:
        stage = "retrieve"
        text = (
            "[SOP stage: retrieve] Call KB_search before quoting a fee, an eligibility rule, or a tool name. "
            "The tool returns the matched clause only. Search again or use grep if that clause is not enough. "
            "Do not call a write tool in this stage."
        )
    elif not verified:
        stage = "verify"
        text = (
            "[SOP stage: verify] Read the customer record and get_current_time, then call log_verification "
            "with those exact identity fields before any account change."
        )
    elif uses_discoverable and not unlocked:
        stage = "unlock"
        text = (
            "[SOP stage: unlock] Call unlock_discoverable_agent_tool before an agent tool named in the retrieved clause. "
            "If the clause says to give a tool to the customer, call give_discoverable_user_tool. "
            "Do not invent a tool name that was not retrieved."
        )
    elif uses_discoverable and not called_inner:
        stage = "fill"
        schemas = "\n".join(_schema_lines(sec))
        text = (
            "[SOP stage: fill] Call the unlocked tool with every required parameter from its schema."
            + (f"\n{schemas}" if schemas else "")
        )
    else:
        stage = "confirm"
        text = (
            "[SOP stage: confirm] Issue the write only with arguments taken from the retrieved clause and the tool schema. "
            "Do not tell the customer the write succeeded before the tool result says so."
        )
    return f"{text} Current stage name: {stage}."


def project_messages_for_model(messages: list[dict], directive: str) -> list[dict]:
    """把阶段指令放进发给模型的副本，不写回会话状态。"""
    if not directive:
        return list(messages)
    copied = [dict(message) for message in messages]
    if copied and copied[0].get("role") == "system":
        copied[0]["content"] = f"{copied[0].get('content') or ''}\n\n{directive}"
        return copied
    return [{"role": "system", "content": directive}, *copied]
